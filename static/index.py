"""
IDEX — Géomarketing · Serveur FastAPI
Import GeoJSON, CSV, Shapefile (.zip)
"""
import io, json, zipfile, tempfile, shutil
from pathlib import Path
import geopandas as gpd
import pandas as pd
import shapely.geometry as sg
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title='IDEX Géomarketing')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

BASE_DIR = Path(__file__).parent
STATIC   = BASE_DIR / 'static'
STATIC.mkdir(exist_ok=True)

# ── Import universel ──────────────────────────────────────────────────────────

def parse_upload(content: bytes, filename: str) -> gpd.GeoDataFrame:
    fn = filename.lower()

    # GeoJSON
    if fn.endswith(('.geojson', '.json')):
        gj = json.loads(content)
        if gj.get('type') == 'FeatureCollection':
            return gpd.GeoDataFrame.from_features(gj['features'], crs='EPSG:4326')
        return gpd.GeoDataFrame.from_features([gj], crs='EPSG:4326')

    # CSV
    if fn.endswith('.csv'):
        df  = pd.read_csv(io.BytesIO(content))
        lat = next((c for c in df.columns if c.lower() in ('lat','latitude','y')), None)
        lon = next((c for c in df.columns if c.lower() in ('lon','lng','longitude','x')), None)
        if not lat or not lon:
            raise HTTPException(400, 'Colonnes lat/lon introuvables dans le CSV')
        return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon], df[lat]), crs='EPSG:4326')

    # Shapefile (.zip contenant .shp + .dbf + .prj etc.)
    if fn.endswith('.zip'):
        tmp = tempfile.mkdtemp()
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                z.extractall(tmp)
            shp_files = list(Path(tmp).rglob('*.shp'))
            if not shp_files:
                raise HTTPException(400, 'Aucun fichier .shp trouvé dans le ZIP')
            gdf = gpd.read_file(str(shp_files[0]))
            if gdf.crs is None:
                gdf = gdf.set_crs('EPSG:4326')
            return gdf.to_crs('EPSG:4326')
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # Shapefile seul (.shp) — rare mais supporté
    if fn.endswith('.shp'):
        tmp = tempfile.mkdtemp()
        try:
            shp_path = Path(tmp) / filename
            shp_path.write_bytes(content)
            gdf = gpd.read_file(str(shp_path))
            return gdf.to_crs('EPSG:4326')
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    raise HTTPException(400, 'Format non supporté. Utilisez .geojson, .csv ou .zip (shapefile)')


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get('/api/health')
def health():
    return {'status': 'ok'}


@app.post('/api/import-points')
async def import_points(file: UploadFile = File(...)):
    """
    Import des points de consommation.
    Retourne un GeoJSON FeatureCollection + stats.
    """
    content  = await file.read()
    filename = file.filename or 'data'
    try:
        gdf = parse_upload(content, filename)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(422, f'Erreur de parsing : {e}')

    # Limiter à 5000 points pour la perf carte
    if len(gdf) > 5000:
        gdf = gdf.head(5000)

    geojson = json.loads(gdf.to_crs('EPSG:4326').to_json())
    bounds  = [round(float(b), 5) for b in gdf.total_bounds]

    return {
        'geojson':       geojson,
        'feature_count': len(gdf),
        'bounds':        bounds,
        'filename':      Path(filename).stem,
        'columns':       [c for c in gdf.columns if c != 'geometry'],
    }


@app.post('/api/import')
async def import_file(file: UploadFile = File(...)):
    """Import générique (dashboard)."""
    content  = await file.read()
    filename = file.filename or 'data'
    try:
        gdf = parse_upload(content, filename)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(422, str(e))

    from profiler import profile
    result  = profile(gdf)
    sample  = gdf.head(2000) if len(gdf) > 2000 else gdf
    geojson = json.loads(sample.to_crs('EPSG:4326').to_json())
    return {**result, 'geojson': geojson, 'filename': Path(filename).stem}


@app.get('/api/demo/{name}')
def get_demo(name: str):
    demos = {'villes': _demo_villes(), 'zones': _demo_zones()}
    if name not in demos:
        raise HTTPException(404, f'Démo inconnue : {name}')
    from profiler import profile
    gdf     = demos[name]
    result  = profile(gdf)
    geojson = json.loads(gdf.to_crs('EPSG:4326').to_json())
    return {**result, 'geojson': geojson, 'filename': f'démo — {name}'}


def _demo_villes():
    data = {
        'name':    ['Paris','Marseille','Lyon','Toulouse','Nice','Nantes','Bordeaux','Lille','Rennes','Montpellier'],
        'pop':     [2161000,861635,515695,471941,340017,314138,254436,232741,216268,285121],
        'region':  ['Île-de-France','PACA','AuRA','Occitanie','PACA','Pays de la Loire','Nouvelle-Aquitaine','Hauts-de-France','Bretagne','Occitanie'],
        'founded': [987,600,43,120,350,70,300,57,52,985],
        'lat':     [48.8566,43.2965,45.764,43.6047,43.7102,47.2184,44.8378,50.6292,48.1173,43.6117],
        'lon':     [2.3522,5.3698,4.8357,1.4442,7.262,-1.5536,-0.5792,3.0573,-1.6778,3.8767],
    }
    df = pd.DataFrame(data)
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lon'], df['lat']), crs='EPSG:4326')

def _demo_zones():
    rows = [
        {'name':'Zone Nord',    'type':'commercial',  'surface_km2':420, 'pop_density':180, 'budget':5200000, 'geometry':sg.box(0.5,47.5,5.5,50.5)},
        {'name':'Zone Sud-Est', 'type':'résidentiel', 'surface_km2':630, 'pop_density':120, 'budget':3800000, 'geometry':sg.box(3.5,42.5,8.5,46.0)},
        {'name':'Zone Ouest',   'type':'agricole',    'surface_km2':500, 'pop_density':45,  'budget':2100000, 'geometry':sg.box(-3.0,43.0,2.0,47.0)},
        {'name':'Zone Centre',  'type':'mixte',       'surface_km2':360, 'pop_density':95,  'budget':4400000, 'geometry':sg.box(1.0,44.0,5.0,47.5)},
        {'name':'Zone Bretagne','type':'touristique', 'surface_km2':270, 'pop_density':70,  'budget':1900000, 'geometry':sg.box(-5.0,47.5,-1.0,49.0)},
    ]
    return gpd.GeoDataFrame(rows, crs='EPSG:4326')


# ── Static ────────────────────────────────────────────────────────────────────

app.mount('/static', StaticFiles(directory=str(STATIC)), name='static')

@app.get('/', response_class=HTMLResponse)
def index():
    p = STATIC / 'index.html'
    return p.read_text() if p.exists() else '<h1>index.html manquant</h1>'
