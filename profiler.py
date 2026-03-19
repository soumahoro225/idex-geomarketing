# ══════════════════════════════════════════════════════
# CELLULE 2 — Profiler (règles Python)
# ══════════════════════════════════════════════════════

from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
import geopandas as gpd
import pandas as pd
import numpy as np

@dataclass
class Widget:
    type: str
    title: str
    priority: int
    span: int = 6
    config: dict = field(default_factory=dict)

# Champs à exclure des histogrammes (identifiants, codes)
ID_PATTERNS = {'id', 'fid', 'objectid', 'gid', 'uid', 'oid',
               'code', 'insee', 'zip', 'cp', 'postal', 'num', 'numero'}

def _is_id_field(name: str) -> bool:
    n = name.lower()
    return any(p in n for p in ID_PATTERNS)

def _is_numeric(s):
    return pd.api.types.is_numeric_dtype(s) and s.nunique() > 1

def _is_year(s):
    if not _is_numeric(s): return False
    vals = s.dropna()
    return bool(len(vals) and vals.between(1800, 2100).all() and s.nunique() <= 200)

def _is_categorical(s, mx=20):
    return s.dtype == object and 1 < s.nunique() <= mx

def _num_stats(s):
    s = s.dropna()
    bins = min(15, max(2, s.nunique()))
    counts, edges = np.histogram(s, bins=bins)
    return {
        'min':    round(float(s.min()), 3),
        'max':    round(float(s.max()), 3),
        'mean':   round(float(s.mean()), 3),
        'median': round(float(s.median()), 3),
        'std':    round(float(s.std()), 3),
        'histogram': {
            'counts': counts.tolist(),
            'edges':  [round(float(e), 3) for e in edges]
        }
    }

def _top_values(s, n=12):
    vc = s.value_counts().head(n)
    total = s.count()
    return [{'value': str(k), 'count': int(v), 'pct': round(v / total * 100, 1)}
            for k, v in vc.items()]

def profile(gdf: gpd.GeoDataFrame) -> dict:
    props  = gdf.drop(columns=['geometry'], errors='ignore')
    gdf_m  = gdf.to_crs('EPSG:3857')
    geom_types = gdf.geometry.geom_type.value_counts().to_dict()
    is_points   = any('Point'   in t for t in geom_types)
    is_polygons = any('Polygon' in t for t in geom_types)
    is_lines    = any('Line'    in t for t in geom_types)
    bounds = [round(float(b), 5) for b in gdf.total_bounds]
    n = len(gdf)

    areas_km2 = None
    if is_polygons:
        a = gdf_m.geometry.area / 1e6
        areas_km2 = {'total': round(float(a.sum()), 3), 'mean': round(float(a.mean()), 3)}

    lengths_km = None
    if is_lines:
        l = gdf_m.geometry.length / 1000
        lengths_km = {'total': round(float(l.sum()), 3), 'mean': round(float(l.mean()), 3)}

    # Profil des champs
    fields = {}
    for col in props.columns:
        s = props[col]
        if _is_year(s):          fields[col] = {'type': 'year',        **_num_stats(s)}
        elif _is_numeric(s):     fields[col] = {'type': 'numeric',     **_num_stats(s)}
        elif _is_categorical(s): fields[col] = {'type': 'categorical', 'top_values': _top_values(s), 'unique_count': int(s.nunique())}
        else:                    fields[col] = {'type': 'text'}

    # Sélection des champs — on exclut les IDs des histogrammes
    num_fields  = [(c, f) for c, f in fields.items()
                   if f['type'] == 'numeric' and not _is_id_field(c)][:4]
    cat_fields  = [(c, f) for c, f in fields.items()
                   if f['type'] == 'categorical' and not _is_id_field(c)][:4]
    year_fields = [(c, f) for c, f in fields.items()
                   if f['type'] == 'year'][:2]

    # KPIs
    kpis = [{'label': 'Entités', 'value': n, 'unit': ''}]
    if areas_km2:
        kpis += [
            {'label': 'Surface totale', 'value': areas_km2['total'], 'unit': 'km²'},
            {'label': 'Surface moyenne', 'value': areas_km2['mean'], 'unit': 'km²'},
        ]
    if lengths_km:
        kpis.append({'label': 'Longueur totale', 'value': lengths_km['total'], 'unit': 'km'})
    kpis.append({'label': 'Champs', 'value': len(props.columns), 'unit': ''})

    # Widgets
    widgets = [
        Widget('map', 'Carte', 1, 8, {'bounds': bounds, 'geom_types': list(geom_types.keys())}),
        Widget('kpi', 'Indicateurs clés', 2, 4, {'kpis': kpis}),
    ]

    for i, (col, f) in enumerate(num_fields):
        widgets.append(Widget('histogram', f'Distribution — {col}', 10 + i, 6, {'field': col, **f}))

    for i, (col, f) in enumerate(cat_fields):
        widgets.append(Widget('bar', f'Répartition — {col}', 20 + i, 6, {'field': col, **f}))

    for i, (col, f) in enumerate(year_fields):
        widgets.append(Widget('timeline', f'Chronologie — {col}', 30 + i, 6, {'field': col, **f}))

    if len(num_fields) >= 2:
        cx, cy = num_fields[0][0], num_fields[1][0]
        pts = props[[cx, cy]].dropna().head(500).rename(columns={cx: 'x', cy: 'y'}).to_dict('records')
        widgets.append(Widget('scatter', f'Corrélation — {cx} vs {cy}', 40, 6,
                              {'field_x': cx, 'field_y': cy, 'points': pts}))

    rows = props.head(60).fillna('').astype(str).to_dict('records')
    widgets.append(Widget('table', 'Aperçu des données', 99, 12,
                          {'columns': list(props.columns), 'rows': rows}))

    widgets.sort(key=lambda w: w.priority)

    return {
        'metadata': {
            'feature_count': n,
            'geometry_types': geom_types,
            'bounds': bounds,
            'field_count': len(props.columns),
            'is_points': is_points,
            'is_polygons': is_polygons,
            'is_lines': is_lines,
        },
        'fields': fields,
        'widgets': [asdict(w) for w in widgets],
    }

print('✓ Profiler prêt — champs ID exclus des histogrammes')
