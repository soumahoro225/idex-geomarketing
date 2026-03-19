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
    """
    Calcule le tracé optimal sur le réseau viaire réel.
    Algorithme :
    1. Télécharge le graphe OSM de la zone (OSMnx)
    2. Applique pénalités sur les arêtes selon les contraintes
    3. Calcule shortest path source → chaque point (Dijkstra)
    4. Construit l'arbre couvrant minimum (MST) pour mutualiser les tronçons
    5. Retourne GeoJSON du tracé
    """
    t0 = time.time()

    src_pt = (req.source['lat'], req.source['lng'])
    dest_pts = [(p['lat'], p['lng']) for p in req.points]

    if not dest_pts:
        raise HTTPException(400, 'Aucun point de destination')

    # Bbox de la zone + buffer
    all_lats = [src_pt[0]] + [p[0] for p in dest_pts]
    all_lngs = [src_pt[1]] + [p[1] for p in dest_pts]
    minlat, maxlat = min(all_lats), max(all_lats)
    minlng, maxlng = min(all_lngs), max(all_lngs)
    buf = 0.003  # ~300m de marge

    # ── 1. Télécharger le graphe OSM ─────────────────────────────────────
    try:
        bbox = (maxlat + buf, minlat - buf, maxlng + buf, minlng - buf)
        G = ox.graph_from_bbox(
            bbox[0], bbox[1], bbox[2], bbox[3],
            network_type='drive',
            simplify=True,
        )
    except Exception as e:
        raise HTTPException(500, f'Erreur chargement OSM : {e}')

    # ── 2. Appliquer pénalités contraintes ───────────────────────────────
    # Charger les contraintes Overpass si activées
    constraint_polys = []
    constraint_lines = []

    if req.constraints.get('water', True) or req.constraints.get('building', True) \
       or req.constraints.get('railway', True) or req.constraints.get('trees', True):
        try:
            import requests as req_lib
            q = f"""[out:json][timeout:20][maxsize:2000000];
            (way["waterway"~"river|stream|canal"]({minlat-buf},{minlng-buf},{maxlat+buf},{maxlng+buf});
             way["natural"="water"]({minlat-buf},{minlng-buf},{maxlat+buf},{maxlng+buf});
             way["railway"~"rail|subway|tram"]({minlat-buf},{minlng-buf},{maxlat+buf},{maxlng+buf});
             way["natural"~"wood|tree_row"]({minlat-buf},{minlng-buf},{maxlat+buf},{maxlng+buf});
             way["landuse"~"forest|wood"]({minlat-buf},{minlng-buf},{maxlat+buf},{maxlng+buf}););
            out geom qt;"""
            r = req_lib.post('https://overpass-api.de/api/interpreter', data=q, timeout=25)
            if r.ok:
                data = r.json()
                for el in data.get('elements', []):
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
                        if tags.get('waterway') or tags.get('natural') == 'water':
                            if req.constraints.get('water', True):
                                constraint_polys.append((poly, 6))
                        elif tags.get('natural') in ('wood','tree_row') or tags.get('landuse') in ('forest','wood'):
                            if req.constraints.get('trees', True):
                                constraint_polys.append((poly, 3))
        except Exception:
            pass  # Les contraintes sont optionnelles

    # Pondérer les arêtes du graphe selon les contraintes
    if constraint_polys or constraint_lines:
        for u, v, k, data in G.edges(data=True, keys=True):
            edge_geom = data.get('geometry')
            if edge_geom is None:
                # Reconstruire depuis les coordonnées des nœuds
                u_data = G.nodes[u]
                v_data = G.nodes[v]
                edge_geom = sg.LineString([(u_data['x'], u_data['y']), (v_data['x'], v_data['y'])])

            penalty = 1.0
            for poly, cost in constraint_polys:
                if edge_geom.intersects(poly):
                    penalty += cost
            for line, cost in constraint_lines:
                if edge_geom.distance(line) < 0.0003:  # ~30m
                    penalty += cost

            base_length = data.get('length', 1)
            G[u][v][k]['weighted_length'] = base_length * penalty

    else:
        # Pas de contraintes : utiliser longueur brute
        for u, v, k, data in G.edges(data=True, keys=True):
            G[u][v][k]['weighted_length'] = data.get('length', 1)

    # ── 3. Trouver les nœuds OSM les plus proches ────────────────────────
    src_node  = ox.nearest_nodes(G, req.source['lng'], req.source['lat'])
    dest_nodes = [ox.nearest_nodes(G, p['lng'], p['lat']) for p in req.points]
    all_nodes = list(set([src_node] + dest_nodes))

    # ── 4. Shortest paths entre tous les nœuds clés ───────────────────────
    # Construire matrice de distances (Dijkstra depuis chaque nœud clé)
    paths = {}
    for node in all_nodes:
        try:
            lengths, path_dict = nx.single_source_dijkstra(
                G, node, weight='weighted_length', cutoff=50000
            )
            paths[node] = (lengths, path_dict)
        except Exception:
            pass

    # ── 5. MST sur le graphe des nœuds clés ──────────────────────────────
    # Construire un graphe complet entre nœuds clés avec distances shortest path
    metric_G = nx.Graph()
    for i, n1 in enumerate(all_nodes):
        for n2 in all_nodes[i+1:]:
            if n1 in paths and n2 in paths[n1][0]:
                d = paths[n1][0][n2]
                metric_G.add_edge(n1, n2, weight=d)
            elif n2 in paths and n1 in paths[n2][0]:
                d = paths[n2][0][n1]
                metric_G.add_edge(n1, n2, weight=d)
            else:
                metric_G.add_edge(n1, n2, weight=999999)

    try:
        mst = nx.minimum_spanning_tree(metric_G, weight='weight')
    except Exception:
        mst = metric_G

    # ── 6. Reconstruire les géométries des arêtes MST ─────────────────────
    features = []
    total_length = 0
    edge_set = set()

    for n1, n2 in mst.edges():
        # Récupérer le chemin réel sur le graphe OSM
        try:
            if n1 in paths and n2 in paths[n1][1]:
                node_path = paths[n1][1][n2]
            elif n2 in paths and n1 in paths[n2][1]:
                node_path = list(reversed(paths[n2][1][n1]))
            else:
                continue

            # Extraire les géométries des arêtes du chemin
            edge_coords = []
            seg_length  = 0
            for i in range(len(node_path) - 1):
                u, v = node_path[i], node_path[i+1]
                edge_key = (min(u,v), max(u,v))

                # Récupérer l'arête (peut y en avoir plusieurs)
                edge_data = G.get_edge_data(u, v) or G.get_edge_data(v, u)
                if not edge_data:
                    continue
                data = edge_data.get(0, list(edge_data.values())[0])

                seg_length += data.get('length', 0)

                if 'geometry' in data:
                    coords = list(data['geometry'].coords)
                    if u != node_path[i] or (i > 0 and coords[0] == edge_coords[-1] if edge_coords else False):
                        coords = coords[::-1]
                else:
                    u_d = G.nodes[u]
                    v_d = G.nodes[v]
                    coords = [(u_d['x'], u_d['y']), (v_d['x'], v_d['y'])]

                if edge_coords and coords and edge_coords[-1] == coords[0]:
                    edge_coords.extend(coords[1:])
                else:
                    edge_coords.extend(coords)

                edge_set.add(edge_key)

            if len(edge_coords) >= 2:
                total_length += seg_length
                features.append({
                    'type': 'Feature',
                    'geometry': {
                        'type': 'LineString',
                        'coordinates': [[round(c[0],6), round(c[1],6)] for c in edge_coords]
                    },
                    'properties': {
                        'length_m': round(seg_length),
                        'from_node': n1,
                        'to_node': n2,
                    }
                })
        except Exception:
            continue

    elapsed = round(time.time() - t0, 1)

    return {
        'geojson': {'type': 'FeatureCollection', 'features': features},
        'stats': {
            'total_length_m': round(total_length),
            'total_length_km': round(total_length / 1000, 2),
            'segments': len(features),
            'points': len(dest_pts),
            'elapsed_s': elapsed,
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
