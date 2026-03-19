"""
IDEX — Réseau de chauffage urbain
Tracé sur réseau viaire réel avec OSMnx + NetworkX
Algorithme : Steiner tree approché via MST sur shortest paths
"""
import io, json, zipfile, tempfile, shutil, time
from pathlib import Path
from typing import List

import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import osmnx as ox
import shapely.geometry as sg
from shapely.ops import unary_union
from fastapi import FastAPI, File, HTTPException, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title='IDEX Réseau Chauffage')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

BASE_DIR = Path(__file__).parent
STATIC   = BASE_DIR / 'static'
STATIC.mkdir(exist_ok=True)

# Cache graphe OSM en mémoire (clé = bbox arrondie)
_graph_cache: dict = {}

# ── Modèles ────────────────────────────────────────────────────────────────

class TraceRequest(BaseModel):
    source: dict          # {lng, lat}
    points: List[dict]    # [{lng, lat}, ...]
    constraints: dict = {}  # {natura: bool, water: bool, railway: bool, building: bool, trees: bool}

class ImportResponse(BaseModel):
    pass

# ── Import fichiers ────────────────────────────────────────────────────────

def parse_upload(content: bytes, filename: str) -> gpd.GeoDataFrame:
    fn = filename.lower()
    if fn.endswith(('.geojson', '.json')):
        gj = json.loads(content)
        if gj.get('type') == 'FeatureCollection':
            return gpd.GeoDataFrame.from_features(gj['features'], crs='EPSG:4326')
        return gpd.GeoDataFrame.from_features([gj], crs='EPSG:4326')
    if fn.endswith('.csv'):
        df  = pd.read_csv(io.BytesIO(content))
        lat = next((c for c in df.columns if c.lower() in ('lat','latitude','y')), None)
        lon = next((c for c in df.columns if c.lower() in ('lon','lng','longitude','x')), None)
        if not lat or not lon:
            raise HTTPException(400, 'Colonnes lat/lon introuvables')
        return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon], df[lat]), crs='EPSG:4326')
    if fn.endswith('.zip'):
        tmp = tempfile.mkdtemp()
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                z.extractall(tmp)
            shp = list(Path(tmp).rglob('*.shp'))
            if not shp:
                raise HTTPException(400, 'Aucun .shp dans le ZIP')
            gdf = gpd.read_file(str(shp[0]))
            if gdf.crs is None:
                gdf = gdf.set_crs('EPSG:4326')
            return gdf.to_crs('EPSG:4326')
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    raise HTTPException(400, 'Format non supporté (.geojson, .csv ou .zip)')


# ── Endpoints import ───────────────────────────────────────────────────────

@app.get('/api/health')
def health():
    return {'status': 'ok'}

@app.post('/api/import-points')
async def import_points(file: UploadFile = File(...)):
    content  = await file.read()
    filename = file.filename or 'data'
    try:
        gdf = parse_upload(content, filename)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(422, str(e))

    if len(gdf) > 5000:
        gdf = gdf.head(5000)

    geojson = json.loads(gdf.to_crs('EPSG:4326').to_json())
    bounds  = [round(float(b), 6) for b in gdf.total_bounds]
    return {
        'geojson':       geojson,
        'feature_count': len(gdf),
        'bounds':        bounds,
        'filename':      Path(filename).stem,
        'columns':       [c for c in gdf.columns if c != 'geometry'],
    }


# ── Endpoint tracé ─────────────────────────────────────────────────────────

@app.post('/api/trace')
async def compute_trace(req: TraceRequest):
    t0 = time.time()

    src_pt  = (req.source['lat'], req.source['lng'])
    all_pts = [(p['lat'], p['lng']) for p in req.points]

    if not all_pts:
        raise HTTPException(400, 'Aucun point de destination')

    # ── Clustering : regrouper les points proches (grille 200m) ──────────
    # Réduit 453 points → ~80-120 clusters, calcul 10x plus rapide
    def cluster_points(pts, grid_deg=0.002):
        clusters = {}
        for lat, lng in pts:
            key = (round(lat / grid_deg) * grid_deg,
                   round(lng / grid_deg) * grid_deg)
            if key not in clusters:
                clusters[key] = []
            clusters[key].append((lat, lng))
        return [
            (sum(c[0] for c in v) / len(v), sum(c[1] for c in v) / len(v))
            for v in clusters.values()
        ]

    dest_pts = cluster_points(all_pts, grid_deg=0.002)

    # Bbox
    all_lats = [src_pt[0]] + [p[0] for p in dest_pts]
    all_lngs = [src_pt[1]] + [p[1] for p in dest_pts]
    minlat, maxlat = min(all_lats), max(all_lats)
    minlng, maxlng = min(all_lngs), max(all_lngs)
    buf = 0.003

    # Clé de cache (bbox arrondie à 2 décimales ~1km)
    cache_key = f"{round(minlat,2)},{round(minlng,2)},{round(maxlat,2)},{round(maxlng,2)}"

    # ── 1. Graphe OSM (cache) ─────────────────────────────────────────────
    if cache_key in _graph_cache:
        G = _graph_cache[cache_key]
    else:
        try:
            G = ox.graph_from_bbox(
                bbox=(maxlat + buf, minlat - buf, maxlng + buf, minlng - buf),
                network_type='drive',
                simplify=True,
            )
            _graph_cache[cache_key] = G
            # Garder max 3 graphes en cache
            if len(_graph_cache) > 3:
                oldest = next(iter(_graph_cache))
                del _graph_cache[oldest]
        except Exception as e:
            raise HTTPException(500, f'Erreur chargement OSM : {e}')

    # ── 2. Pénalités contraintes ──────────────────────────────────────────
    import requests as req_lib

    constraint_polys = []
    constraint_lines = []

    if any(req.constraints.get(k, True) for k in ['water','building','railway','trees']):
        try:
            q = f"""[out:json][timeout:20][maxsize:2000000];
            (way["waterway"~"river|stream|canal"]({minlat-buf},{minlng-buf},{maxlat+buf},{maxlng+buf});
             way["natural"="water"]({minlat-buf},{minlng-buf},{maxlat+buf},{maxlng+buf});
             way["railway"~"rail|subway|tram"]({minlat-buf},{minlng-buf},{maxlat+buf},{maxlng+buf});
             way["natural"~"wood|tree_row"]({minlat-buf},{minlng-buf},{maxlat+buf},{maxlng+buf});
             way["landuse"~"forest|wood"]({minlat-buf},{minlng-buf},{maxlat+buf},{maxlng+buf}););
            out geom qt;"""
            r = req_lib.post('https://overpass-api.de/api/interpreter', data=q, timeout=15)
            if r.ok:
                for el in r.json().get('elements', []):
                    tags = el.get('tags', {})
                    geom = el.get('geometry', [])
                    if not geom:
                        continue
                    coords = [(g['lon'], g['lat']) for g in geom]
                    if tags.get('railway') and req.constraints.get('railway', True):
                        if len(coords) >= 2:
                            constraint_lines.append((sg.LineString(coords), 8))
                    elif len(coords) >= 3:
                        closed = coords + [coords[0]] if coords[0] != coords[-1] else coords
                        poly = sg.Polygon(closed)
                        if not poly.is_valid:
                            poly = poly.buffer(0)
                        if (tags.get('waterway') or tags.get('natural') == 'water') and req.constraints.get('water', True):
                            constraint_polys.append((poly, 6))
                        elif (tags.get('natural') in ('wood', 'tree_row') or tags.get('landuse') in ('forest', 'wood')) and req.constraints.get('trees', True):
                            constraint_polys.append((poly, 3))
        except Exception:
            pass

    # Pondérer les arêtes
    for u, v, k, data in G.edges(data=True, keys=True):
        edge_geom = data.get('geometry')
        if edge_geom is None:
            u_d, v_d = G.nodes[u], G.nodes[v]
            edge_geom = sg.LineString([(u_d['x'], u_d['y']), (v_d['x'], v_d['y'])])
        penalty = 1.0
        for poly, cost in constraint_polys:
            if edge_geom.intersects(poly):
                penalty += cost
        for line, cost in constraint_lines:
            if edge_geom.distance(line) < 0.0003:
                penalty += cost
        G[u][v][k]['weighted_length'] = data.get('length', 1) * penalty

    # ── 3. Nœuds les plus proches ─────────────────────────────────────────
    src_node   = ox.nearest_nodes(G, req.source['lng'], req.source['lat'])

    # Mapper chaque cluster → nœud OSM + coordonnées réelles du point
    dest_map = []  # [(dest_node, real_lng, real_lat), ...]
    for lat, lng in dest_pts:
        node = ox.nearest_nodes(G, lng, lat)
        dest_map.append((node, lng, lat))

    dest_nodes = list(set(d[0] for d in dest_map))

    # ── 4. Shortest Path Tree depuis la source ────────────────────────────
    try:
        lengths, paths = nx.single_source_dijkstra(
            G, src_node, weight='weighted_length', cutoff=50000
        )
    except Exception as e:
        raise HTTPException(500, f'Erreur Dijkstra : {e}')

    # ── 5. Union des chemins → tronçons mutualisés ─────────────────────────
    used_edges = {}

    for dest_node, real_lng, real_lat in dest_map:
        if dest_node not in paths:
            continue
        node_path = paths[dest_node]
        for i in range(len(node_path) - 1):
            u, v = node_path[i], node_path[i + 1]
            edge_key = (min(u, v), max(u, v))
            if edge_key in used_edges:
                continue
            edge_data = G.get_edge_data(u, v) or G.get_edge_data(v, u)
            if not edge_data:
                continue
            data = edge_data.get(0, list(edge_data.values())[0])
            if 'geometry' in data:
                coords = list(data['geometry'].coords)
                u_x = G.nodes[u]['x']
                if len(coords) > 1 and abs(coords[-1][0] - u_x) < abs(coords[0][0] - u_x):
                    coords = coords[::-1]
            else:
                coords = [(G.nodes[u]['x'], G.nodes[u]['y']),
                          (G.nodes[v]['x'], G.nodes[v]['y'])]
            used_edges[edge_key] = {
                'coords': coords,
                'length': data.get('length', 0),
                'is_connector': False,
            }

        # Segment final : nœud OSM → point de consommation réel
        node_x = G.nodes[dest_node]['x']
        node_y = G.nodes[dest_node]['y']
        conn_key = f'conn_{dest_node}_{round(real_lng,6)}_{round(real_lat,6)}'
        if conn_key not in used_edges:
            import math
            dist_m = math.sqrt((node_x - real_lng)**2 + (node_y - real_lat)**2) * 111000
            used_edges[conn_key] = {
                'coords': [(node_x, node_y), (real_lng, real_lat)],
                'length': dist_m,
                'is_connector': True,
            }

    # ── 6. GeoJSON ────────────────────────────────────────────────────────
    features     = []
    total_length = 0

    for edata in used_edges.values():
        coords = edata['coords']
        length = edata['length']
        if len(coords) >= 2:
            total_length += length
            features.append({
                'type': 'Feature',
                'geometry': {
                    'type': 'LineString',
                    'coordinates': [[round(c[0], 6), round(c[1], 6)] for c in coords]
                },
                'properties': {
                    'length_m':     round(length),
                    'is_connector': edata.get('is_connector', False),
                }
            })

    elapsed = round(time.time() - t0, 1)

    return {
        'geojson': {'type': 'FeatureCollection', 'features': features},
        'stats': {
            'total_length_m':  round(total_length),
            'total_length_km': round(total_length / 1000, 2),
            'segments':        len(features),
            'points':          len(all_pts),
            'clusters':        len(dest_pts),
            'elapsed_s':       elapsed,
        }
    }


# ── Demo data ──────────────────────────────────────────────────────────────

@app.get('/api/demo/{name}')
def get_demo(name: str):
    demos = {'villes': _demo_villes(), 'zones': _demo_zones()}
    if name not in demos:
        raise HTTPException(404)
    try:
        from profiler import profile
        gdf    = demos[name]
        result = profile(gdf)
        geojson = json.loads(gdf.to_crs('EPSG:4326').to_json())
        return {**result, 'geojson': geojson, 'filename': f'démo — {name}'}
    except Exception:
        raise HTTPException(500, 'Profiler non disponible')

def _demo_villes():
    data = {
        'name':    ['Paris','Marseille','Lyon','Toulouse','Nice','Nantes','Bordeaux','Lille','Rennes','Montpellier'],
        'pop':     [2161000,861635,515695,471941,340017,314138,254436,232741,216268,285121],
        'region':  ['Île-de-France','PACA','AuRA','Occitanie','PACA','Pays de la Loire','Nouvelle-Aquitaine','Hauts-de-France','Bretagne','Occitanie'],
        'lat':     [48.8566,43.2965,45.764,43.6047,43.7102,47.2184,44.8378,50.6292,48.1173,43.6117],
        'lon':     [2.3522,5.3698,4.8357,1.4442,7.262,-1.5536,-0.5792,3.0573,-1.6778,3.8767],
    }
    df = pd.DataFrame(data)
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lon'], df['lat']), crs='EPSG:4326')

def _demo_zones():
    rows = [
        {'name':'Zone Nord','type':'commercial','surface_km2':420,'geometry':sg.box(0.5,47.5,5.5,50.5)},
        {'name':'Zone Sud-Est','type':'résidentiel','surface_km2':630,'geometry':sg.box(3.5,42.5,8.5,46.0)},
    ]
    return gpd.GeoDataFrame(rows, crs='EPSG:4326')


# ── Static ─────────────────────────────────────────────────────────────────

app.mount('/static', StaticFiles(directory=str(STATIC)), name='static')

@app.get('/', response_class=HTMLResponse)
def index():
    p = STATIC / 'index.html'
    return p.read_text() if p.exists() else '<h1>index.html manquant</h1>'
