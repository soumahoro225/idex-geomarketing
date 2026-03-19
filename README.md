# IDEX — Géomarketing

Dashboard géospatial adaptatif — le dashboard se génère automatiquement selon vos données.

## Déployer sur Render (gratuit)

1. Poussez ce repo sur GitHub
2. [render.com](https://render.com) → **New Web Service** → connectez le repo
3. Render détecte `render.yaml` → **Deploy**
4. URL : `https://idex-geomarketing.onrender.com`

## Tester en local

```bash
python start.py
# → http://localhost:8000
```

## Structure

```
idex-geomarketing/
├── server.py          # FastAPI
├── profiler.py        # Moteur de règles Python
├── static/index.html  # Frontend
├── start.py           # Lancement local
├── requirements.txt
├── Dockerfile
├── render.yaml
└── .gitignore
```

## Formats supportés

- **GeoJSON** `.geojson` `.json`
- **CSV** avec colonnes `lat` / `lon`
