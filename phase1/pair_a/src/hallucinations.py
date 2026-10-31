"""Module for Idea 4: False Detection Gallery & Empty Image Hallucinations.

Provides:
- Visual background characteristic extraction (luminance, greenness, glare, IR flash).
- Multi-model comparative analysis (WILDS Fine-Tuned ResNet-50 vs. ImageNet Linear Probe).
- Top-5 class probability extraction on false detection frames.
- Camera burst persistence & sequence dynamics.
- Interactive Camera Hotspots Bar Chart (clickable to filter gallery).
- Phantom Species Breakdown & Taxonomic Ranking table.
- Interactive Confusion Matrix (Background x Species Hallucinations).
- Multi-Frame Photo Burst Sequence Viewer & Case Study.
- Standalone interactive light-themed HTML False Detection Gallery.
"""
from pathlib import Path
import base64
import io
import json

import numpy as np
import pandas as pd
from PIL import Image

from common import ROOT, resolve, load_config
from common.data import load_class_names


def extract_background_features(df: pd.DataFrame, image_dir: Path) -> pd.DataFrame:
    """Compute visual background metrics directly from image pixel arrays.

    The iWildCam dataset contains no environmental annotations. This function
    derives 4 empirical background categories from colorimetry and remote sensing:
      - 'Night / IR Flash': Monochrome frames (color difference < 12) or dark scenes (mean lum < 60).
      - 'Dense Vegetation': High chlorophyll content (Excess Green Index ExG = 2G - (R + B) > 20).
      - 'High Contrast / Lighting': Harsh sun glare (>5% pixels > 240) or high variance (std lum > 65).
      - 'Open / Diffuse Landscape': Balanced, diffuse daytime trail/clearing scenes with low greenness.
    """
    image_dir = Path(image_dir)
    features = []

    for _, row in df.iterrows():
        img_path = image_dir / row['filename']
        if not img_path.exists():
            continue

        with Image.open(img_path) as img:
            rgb = np.array(img.convert('RGB'), dtype=np.float32)

        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        mean_lum = float(np.mean(lum))
        std_lum = float(np.std(lum))

        color_diff = float(np.mean(np.abs(r - g) + np.abs(g - b) + np.abs(b - r)))
        exg = 2.0 * g - (r + b)
        greenness = float(np.mean(exg))
        glare_frac = float(np.mean(lum > 240.0))

        is_night = (mean_lum < 60.0) or (color_diff < 12.0)

        if is_night:
            bg_category = 'Night / IR Flash'
        elif greenness > 20.0:
            bg_category = 'Dense Vegetation'
        elif glare_frac > 0.05 or std_lum > 65.0:
            bg_category = 'High Contrast / Lighting'
        else:
            bg_category = 'Open / Diffuse Landscape'

        features.append({
            'id': row['id'],
            'mean_luminance': round(mean_lum, 2),
            'std_luminance': round(std_lum, 2),
            'color_diff': round(color_diff, 2),
            'greenness': round(greenness, 2),
            'glare_fraction': round(glare_frac, 4),
            'is_night': bool(is_night),
            'bg_category': bg_category,
        })

    feat_df = pd.DataFrame(features)
    return df.merge(feat_df, on='id', how='left')


def load_comparative_data(split: str = 'val', compute_bg_features: bool = True):
    """Load model predictions and extract false detection and disagreement subsets.

    Reads precomputed predictions from the WILDS Fine-Tuned ResNet-50 and
    ImageNet Linear Probe baselines. Filters for ground-truth empty frames
    (label == 0) and identifies false animal detections (prediction != 0).
    Optionally enriches frames with visual background features.
    """
    ft_path = resolve('outputs/predictions.parquet')
    pr_path = resolve('outputs/predictions_probe.parquet')

    if not ft_path.exists() or not pr_path.exists():
        raise FileNotFoundError("Prediction files missing in outputs/. Run `make inference` and `make baselines` first.")

    df_ft = pd.read_parquet(ft_path)
    df_pr = pd.read_parquet(pr_path)

    if split != 'all':
        df_ft = df_ft[df_ft['split'] == split].copy()
        df_pr = df_pr[df_pr['split'] == split].copy()

    cfg = load_config()
    class_names = load_class_names(cfg)

    true_empty = df_ft[df_ft['label'] == 0].copy()

    # False animal detection (y=0, pred!=0)
    ft_fp = df_ft[(df_ft['label'] == 0) & (df_ft['prediction'] != 0)].copy()
    probe_fp = df_pr[(df_pr['label'] == 0) & (df_pr['prediction'] != 0)].copy()

    # Missed animal (y!=0, pred==0)
    ft_fn = df_ft[(df_ft['label'] != 0) & (df_ft['prediction'] == 0)].copy()
    probe_fn = df_pr[(df_pr['label'] != 0) & (df_pr['prediction'] == 0)].copy()

    # Model comparisons on empty images
    merged_empty = pd.DataFrame({
        'id': true_empty['id'],
        'location': true_empty['location'],
        'seq_id': true_empty['seq_id'],
        'datetime': true_empty['datetime'],
        'filename': true_empty['filename'],
        'ft_pred': df_ft.loc[true_empty.index, 'prediction'],
        'ft_pred_name': df_ft.loc[true_empty.index, 'prediction_name'],
        'ft_conf': df_ft.loc[true_empty.index, 'confidence'],
        'probe_pred': df_pr.loc[true_empty.index, 'prediction'],
        'probe_pred_name': df_pr.loc[true_empty.index, 'prediction_name'],
        'probe_conf': df_pr.loc[true_empty.index, 'confidence'],
    })

    merged_empty['ft_error'] = merged_empty['ft_pred'] != 0
    merged_empty['probe_error'] = merged_empty['probe_pred'] != 0

    merged_empty['comparison'] = np.select(
        [
            merged_empty['ft_error'] & merged_empty['probe_error'],
            merged_empty['ft_error'] & ~merged_empty['probe_error'],
            ~merged_empty['ft_error'] & merged_empty['probe_error'],
        ],
        ['both_hallucinated', 'ft_only_hallucinated', 'probe_only_hallucinated'],
        default='both_correct_empty',
    )

    disagreements = merged_empty[merged_empty['comparison'].isin(['ft_only_hallucinated', 'probe_only_hallucinated'])].copy()

    if compute_bg_features:
        img_dir = resolve('data/iwildcam_v2.0/train')
        ft_fp = extract_background_features(ft_fp, img_dir)
        probe_fp = extract_background_features(probe_fp, img_dir)
        disagreements = extract_background_features(disagreements, img_dir)

    return {
        'ft_fp': ft_fp,
        'probe_fp': probe_fp,
        'ft_fn': ft_fn,
        'probe_fn': probe_fn,
        'disagreements': disagreements,
        'true_empty': true_empty,
        'full_ft': df_ft,
        'full_probe': df_pr,
        'class_names': class_names,
    }


def get_top5_distribution(row: pd.Series, class_names: pd.Series) -> list[dict]:
    """Extract top 5 predicted classes with probabilities from a prediction row."""
    prob_cols = [f'prob_{i}' for i in range(len(class_names))]
    probs = row[prob_cols].to_numpy(dtype=float)
    top5_idx = np.argsort(probs)[::-1][:5]
    return [
        {
            'rank': r + 1,
            'class_id': int(idx),
            'species': str(class_names[idx]),
            'prob': float(probs[idx]),
            'prob_pct': f"{probs[idx]:.1%}",
            'is_empty': bool(idx == 0),
        }
        for r, idx in enumerate(top5_idx)
    ]


def analyze_burst_persistence(fp_df: pd.DataFrame, empty_df: pd.DataFrame) -> dict:
    """Analyze temporal consistency of hallucinations across camera burst sequences."""
    empty_seqs = empty_df.groupby(['location', 'seq_id']).size().rename('seq_length')
    fp_seqs = fp_df.groupby(['location', 'seq_id']).agg(
        fp_count=('id', 'size'),
        species_variety=('prediction_name', 'nunique'),
        top_species=('prediction_name', lambda s: s.mode()[0] if not s.empty else 'none'),
    )

    merged = pd.concat([empty_seqs, fp_seqs], axis=1).fillna({'fp_count': 0, 'species_variety': 0})
    affected = merged[merged['fp_count'] > 0].copy()
    affected['fp_fraction'] = affected['fp_count'] / affected['seq_length']

    return {
        'total_empty_sequences': len(merged),
        'affected_sequences': len(affected),
        'fully_hallucinated_sequences': int((affected['fp_count'] == affected['seq_length']).sum()),
        'full_hallucination_rate': float((affected['fp_count'] == affected['seq_length']).mean()),
        'avg_fps_per_affected_burst': float(affected['fp_count'].mean()),
        'sequences_df': affected.sort_values('fp_count', ascending=False).reset_index(),
    }


def generate_thumbnail_base64(image_path: Path, max_size=(280, 280)) -> str:
    """Create a lightweight base64 JPEG thumbnail string."""
    with Image.open(image_path) as img:
        img.thumbnail(max_size)
        buffer = io.BytesIO()
        img.convert('RGB').save(buffer, format='JPEG', quality=82)
        return 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode('utf-8')


def build_false_detection_gallery_html(
    data_bundle: dict,
    output_path: Path = None,
    image_dir: Path = None,
) -> str:
    """Build the standalone, interactive light-themed False Detection Gallery dashboard.

    Integrates:
      1. Top Educational Explainer with 4 clickable thematic discovery cards.
      2. Problem Cameras Bar Chart (with error-rate and count modes, camera filtering).
      3. Phantom Species Ranking Table (with binomial scientific names and emojis).
      4. Environment Triggers Matrix (Background x Species cross-tabulation and inspector).
      5. Photo Bursts Sequence Dynamics (multi-shot temporal persistence & case study).
      6. Filterable Gallery Grid with dynamic pagination, confidence thresholding, and modal view.
    """
    if output_path is None:
        output_path = resolve('phase1/pair_a/figures/false_detection_gallery.html')
    if image_dir is None:
        image_dir = resolve('data/iwildcam_v2.0/train')

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ft_fp = data_bundle['ft_fp']
    probe_fp = data_bundle['probe_fp']
    true_empty = data_bundle['true_empty']
    class_names = data_bundle['class_names']

    def format_scientific_name(sp: str) -> str:
        """Format biological species names in standard binomial nomenclature (e.g. Panthera onca)."""
        sp_clean = sp.strip().replace('_', ' ')
        if not sp_clean:
            return sp
        parts = sp_clean.split()
        if len(parts) >= 2:
            return parts[0].capitalize() + ' ' + ' '.join(p.lower() for p in parts[1:])
        return sp_clean.capitalize()

    SPECIES_EMOJIS = {
        'panthera onca': '🐆',
        'leopardus pardalis': '🐆',
        'crax rubra': '🦃',
        'equus ferus': '🐎',
        'dasyprocta punctata': '🦔',
        'meleagris ocellata': '🦃',
        'madoqua guentheri': '🦌',
        'urocyon cinereoargenteus': '🦊',
        'motorcycle': '🏍️',
        'argusianus argus': '🦚',
        'aepyceros melampus': '🦌',
        'giraffa camelopardalis': '🦒',
        'tapirus bairdii': '🦛',
        'bos taurus': '🐂',
        'loxodonta africana': '🐘',
        'paguma larvata': '🦝',
        'crocuta crocuta': '🐕',
        'panthera tigris': '🐅',
        'nasua narica': '🦝',
        'tayassu pecari': '🐗',
        'helarctos malayanus': '🐻',
        'lepus saxatilis': '🐇',
        'equus quagga': '🦓',
        'tinamus major': '🐦',
        'agouti paca': '🦔',
        'mazama temama': '🦌',
        'funisciurus carruthersi': '🐿️',
        'equus grevyi': '🦓',
    }

    # Process fine-tuned records
    ft_items = []
    for _, row in ft_fp.iterrows():
        img_path = image_dir / row['filename']
        thumb = generate_thumbnail_base64(img_path) if img_path.exists() else ''
        top5 = get_top5_distribution(row, class_names)
        sp_name = row['prediction_name']

        ft_items.append({
            'id': row['id'],
            'filename': row['filename'],
            'thumb': thumb,
            'model': 'Fine-Tuned ResNet-50',
            'species': sp_name,
            'scientific_name': format_scientific_name(sp_name),
            'emoji': SPECIES_EMOJIS.get(sp_name, '🐾'),
            'confidence': round(float(row['confidence']), 3),
            'confidence_pct': f"{row['confidence']:.1%}",
            'camera': str(row['location']),
            'seq_id': str(row.get('seq_id', '')),
            'datetime': str(row.get('datetime', '')),
            'bg': row.get('bg_category', 'Unknown'),
            'luminance': row.get('mean_luminance', 0),
            'greenness': row.get('greenness', 0),
            'contrast': row.get('std_luminance', 0),
            'top5': top5,
        })

    # Process probe records
    probe_items = []
    for _, row in probe_fp.iterrows():
        img_path = image_dir / row['filename']
        thumb = generate_thumbnail_base64(img_path) if img_path.exists() else ''
        top5 = get_top5_distribution(row, class_names)
        sp_name = row['prediction_name']

        probe_items.append({
            'id': row['id'],
            'filename': row['filename'],
            'thumb': thumb,
            'model': 'ImageNet Linear Probe',
            'species': sp_name,
            'scientific_name': format_scientific_name(sp_name),
            'emoji': SPECIES_EMOJIS.get(sp_name, '🐾'),
            'confidence': round(float(row['confidence']), 3),
            'confidence_pct': f"{row['confidence']:.1%}",
            'camera': str(row['location']),
            'seq_id': str(row.get('seq_id', '')),
            'datetime': str(row.get('datetime', '')),
            'bg': row.get('bg_category', 'Unknown'),
            'luminance': row.get('mean_luminance', 0),
            'greenness': row.get('greenness', 0),
            'contrast': row.get('std_luminance', 0),
            'top5': top5,
        })

    # 1. Camera statistics & bar chart data
    cam_empty = true_empty.groupby('location').size()
    cam_total = data_bundle['full_ft'].groupby('location').size()
    cam_fp = ft_fp.groupby('location').agg(
        n_fp=('id', 'size'),
        mean_conf=('confidence', 'mean'),
        top_species=('prediction_name', lambda s: s.mode()[0]),
        bg_mode=('bg_category', lambda s: s.mode()[0]),
        rep_filename=('filename', 'first')
    )
    cam_merged = cam_fp.join(cam_empty.rename('n_empty')).join(cam_total.rename('n_total')).sort_values('n_fp', ascending=False)
    cam_merged['fpr'] = cam_merged['n_fp'] / cam_merged['n_empty']

    camera_bars = []
    for loc, r in cam_merged.iterrows():
        rep_path = image_dir / r['rep_filename']
        rep_thumb = generate_thumbnail_base64(rep_path, max_size=(160, 120)) if rep_path.exists() else ''
        n_tot = int(r['n_total']) if pd.notnull(r.get('n_total')) else int(r['n_empty'])
        camera_bars.append({
            'camera': str(loc),
            'n_fp': int(r['n_fp']),
            'n_empty': int(r['n_empty']),
            'n_total': n_tot,
            'fpr_pct': f"{r['fpr']:.1%}",
            'fpr_val': round(float(r['fpr']), 4),
            'overall_fpr_pct': f"{r['n_fp'] / n_tot:.2%}" if n_tot > 0 else '0.0%',
            'top_species': r['top_species'],
            'bg': r['bg_mode'],
            'mean_conf': f"{r['mean_conf']:.1%}",
            'thumb': rep_thumb,
        })

    # 2. RQ3 Background x Species Confusion Matrix data
    top_spp = [
        'panthera onca',
        'leopardus pardalis',
        'crax rubra',
        'equus ferus',
        'meleagris ocellata',
        'urocyon cinereoargenteus',
        'dasyprocta punctata',
        'madoqua guentheri',
        'aepyceros melampus',
        'bos taurus',
    ]
    bg_cats = ['Night / IR Flash', 'Dense Vegetation', 'High Contrast / Lighting', 'Open / Diffuse Landscape']

    def build_matrix_data(df_fp):
        df_copy = df_fp.copy()
        df_copy['sp_cat'] = df_copy['prediction_name'].apply(lambda s: s if s in top_spp else 'other species')
        ct = pd.crosstab(df_copy['bg_category'], df_copy['sp_cat'])
        cols = top_spp + ['other species']
        ct = ct.reindex(index=bg_cats, columns=cols).fillna(0).astype(int)

        rows = []
        for bg in bg_cats:
            cells = []
            bg_total = int(ct.loc[bg].sum()) if bg in ct.index else 0
            for sp in cols:
                count = int(ct.loc[bg, sp]) if (bg in ct.index and sp in ct.columns) else 0
                pct = count / bg_total if bg_total > 0 else 0.0
                cells.append({
                    'species': sp,
                    'count': count,
                    'pct_str': f"{pct:.1%}",
                    'pct_val': round(pct, 3),
                })
            rows.append({
                'bg': bg,
                'total': bg_total,
                'cells': cells,
            })
        return {'species': cols, 'rows': rows}

    matrix_data = {
        'fine_tuned': build_matrix_data(ft_fp),
        'probe': build_matrix_data(probe_fp),
    }

    # 3. Burst persistence & sequence dynamics
    burst_res = analyze_burst_persistence(ft_fp, true_empty)
    burst_summary = {
        'affected_sequences': int(burst_res['affected_sequences']),
        'fully_hallucinated_sequences': int(burst_res['fully_hallucinated_sequences']),
        'full_hallucination_rate': f"{burst_res['full_hallucination_rate']:.1%}",
        'avg_fps': f"{burst_res['avg_fps_per_affected_burst']:.1f}",
    }

    # 5. Species Hallucination Frequency & Diagnostics
    def build_species_stats(df_fp):
        if len(df_fp) == 0:
            return []
        total_fp = len(df_fp)
        records = []
        for sp, group in df_fp.groupby('prediction_name'):
            count = len(group)
            pct = count / total_fp
            mean_conf = group['confidence'].mean()
            median_conf = group['confidence'].median()
            top_bg = group['bg_category'].mode()[0] if 'bg_category' in group and len(group['bg_category']) > 0 else 'Unknown'
            bg_pct = (group['bg_category'] == top_bg).sum() / count if count > 0 else 0.0
            top_cam = group['location'].mode()[0] if 'location' in group and len(group['location']) > 0 else 'Unknown'
            records.append({
                'species': sp,
                'scientific_name': format_scientific_name(sp),
                'emoji': SPECIES_EMOJIS.get(sp, '🐾'),
                'count': int(count),
                'pct_val': round(float(pct), 4),
                'pct_str': f"{pct:.1%}",
                'mean_conf_val': round(float(mean_conf), 4),
                'mean_conf_str': f"{mean_conf:.1%}",
                'median_conf_str': f"{median_conf:.1%}",
                'top_bg': top_bg,
                'top_bg_pct': f"{bg_pct:.0%}",
                'top_camera': str(top_cam),
            })
        records.sort(key=lambda r: r['count'], reverse=True)
        return records

    species_data = {
        'fine_tuned': build_species_stats(ft_fp),
        'probe': build_species_stats(probe_fp),
    }

    all_data = {
        'fine_tuned': ft_items,
        'probe': probe_items,
        'cameras': camera_bars,
        'species': species_data,
        'matrix': matrix_data,
        'burst': burst_summary,
    }

    species_list = sorted(list(set([it['species'] for it in ft_items + probe_items])))
    camera_list = [c['camera'] for c in camera_bars]
    bg_list = sorted(list(set([it['bg'] for it in ft_items + probe_items])))

    species_options = "\n".join([
        f'<option value="{s}">{SPECIES_EMOJIS.get(s, "🐾")} {format_scientific_name(s)}</option>'
        for s in species_list
    ])
    camera_options = "\n".join([f'<option value="{c}">Camera {c}</option>' for c in camera_list])
    bg_options = "\n".join([f'<option value="{b}">{b}</option>' for b in bg_list])
    data_json = json.dumps(all_data)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Camera-Trap False Detection Gallery & Audit Dashboard</title>
  <style>
    :root {{
      --bg: #f8fafc;
      --card-bg: #ffffff;
      --card-hover: #f1f5f9;
      --border: #e2e8f0;
      --border-dark: #cbd5e1;
      --border-active: #2563eb;
      --text: #0f172a;
      --text-muted: #475569;
      --text-dim: #94a3b8;
      --primary: #2563eb;
      --primary-light: #eff6ff;
      --accent-green: #059669;
      --accent-amber: #d97706;
      --accent-rose: #dc2626;
      --accent-purple: #7c3aed;
      --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
      --shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px -1px rgba(0, 0, 0, 0.1);
      --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
      --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.1);
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      padding: 24px;
      line-height: 1.5;
    }}
    
    /* Top Header */
    header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 20px;
      flex-wrap: wrap;
      gap: 16px;
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 20px 24px;
      box-shadow: var(--shadow-sm);
    }}
    h1 {{
      font-size: 24px;
      font-weight: 800;
      color: var(--text);
      letter-spacing: -0.02em;
    }}
    p.sub {{
      color: var(--text-muted);
      font-size: 13px;
      margin-top: 4px;
    }}
    
    /* Stat Pills */
    .stat-pills {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .pill {{
      background: #f8fafc;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 8px 12px;
      font-size: 11px;
      display: flex;
      flex-direction: column;
      gap: 2px;
      box-shadow: var(--shadow-sm);
    }}
    .pill .pill-label {{
      color: var(--text-dim);
      font-size: 10px;
      text-transform: uppercase;
      font-weight: 700;
      letter-spacing: 0.04em;
    }}
    .pill .pill-val {{
      font-size: 14px;
      font-weight: 800;
      color: var(--text);
    }}
    .pill .pill-sub {{
      font-size: 10px;
      color: var(--text-muted);
    }}

    /* Educational Explainer Card */
    .explainer-card {{
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 20px 24px;
      margin-bottom: 22px;
      box-shadow: var(--shadow-sm);
    }}
    .explainer-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 12px;
    }}
    .explainer-title {{
      font-size: 16px;
      font-weight: 800;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .explainer-pill {{
      font-size: 11px;
      font-weight: 700;
      background: #eff6ff;
      color: #1e40af;
      padding: 4px 12px;
      border-radius: 9999px;
      border: 1px solid #bfdbfe;
    }}
    .explainer-body {{
      font-size: 13px;
      color: var(--text-muted);
      line-height: 1.6;
    }}
    .explainer-features {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 12px;
      margin-top: 14px;
    }}
    .feature-item {{
      background: #f8fafc;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px 14px;
      font-size: 12px;
      color: var(--text);
      line-height: 1.45;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }}
    .feature-item:hover {{
      background: #ffffff;
      transform: translateY(-2px);
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }}
    .feature-blue {{ border-left: 4px solid #2563eb; }}
    .feature-blue:hover {{ border-color: #93c5fd; border-left: 4px solid #2563eb; }}
    .feature-rose {{ border-left: 4px solid #e11d48; }}
    .feature-rose:hover {{ border-color: #fca5a5; border-left: 4px solid #e11d48; }}
    .feature-green {{ border-left: 4px solid #059669; }}
    .feature-green:hover {{ border-color: #86efac; border-left: 4px solid #059669; }}
    .feature-purple {{ border-left: 4px solid #7c3aed; }}
    .feature-purple:hover {{ border-color: #ddd6fe; border-left: 4px solid #7c3aed; }}
    .feature-link {{
      margin-top: 8px;
      font-size: 11px;
      font-weight: 700;
      display: inline-flex;
      align-items: center;
      gap: 4px;
      opacity: 0.85;
    }}
    .feature-item:hover .feature-link {{
      opacity: 1;
      text-decoration: underline;
    }}

    /* Active Feature Card Styling */
    .feature-item.active {{
      background: #ffffff;
      box-shadow: 0 4px 14px rgba(0, 0, 0, 0.09);
      transform: translateY(-2px);
    }}
    .feature-item.active.feature-blue {{ border-color: #2563eb; background: #eff6ff; }}
    .feature-item.active.feature-rose {{ border-color: #e11d48; background: #fff1f2; }}
    .feature-item.active.feature-green {{ border-color: #059669; background: #f0fdf4; }}
    .feature-item.active.feature-purple {{ border-color: #7c3aed; background: #f5f3ff; }}

    /* Tab Content Containers */
    .tab-pane {{
      display: none;
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 22px;
      margin-bottom: 24px;
      box-shadow: var(--shadow-sm);
    }}
    .tab-pane.active {{
      display: block;
    }}
    #pane-chart {{ border-top: 5px solid #2563eb; }}
    #pane-species {{ border-top: 5px solid #e11d48; }}
    #pane-matrix {{ border-top: 5px solid #059669; }}
    #pane-burst {{ border-top: 5px solid #7c3aed; }}
    .pane-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .pane-title {{
      font-size: 15px;
      font-weight: 700;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .pane-badge {{
      font-size: 11px;
      font-weight: 600;
      background: var(--primary-light);
      color: var(--primary);
      padding: 3px 10px;
      border-radius: 9999px;
      border: 1px solid rgba(37,99,235,0.2);
    }}
    .pane-desc {{
      font-size: 12px;
      color: var(--text-muted);
      margin-top: 4px;
    }}

    /* Ranked Species List */
    .species-row {{
      display: grid;
      grid-template-columns: 45px 1fr 140px 140px 180px 110px;
      gap: 12px;
      padding: 12px 16px;
      border-bottom: 1px solid #f1f5f9;
      align-items: center;
      cursor: pointer;
      transition: all 0.2s ease;
    }}
    .species-row:last-child {{
      border-bottom: none;
    }}
    .species-row:hover {{
      background: #fff1f2;
    }}
    .species-filter-btn {{
      font-size: 11px;
      font-weight: 700;
      color: #e11d48;
      background: #ffffff;
      border: 1px solid #fecdd3;
      padding: 4px 10px;
      border-radius: 6px;
      transition: all 0.2s ease;
      white-space: nowrap;
    }}
    .species-row:hover .species-filter-btn {{
      background: #e11d48;
      color: #ffffff;
      border-color: #e11d48;
    }}

    /* Interactive Camera Bar Chart */
    .chart-wrap {{
      display: flex;
      align-items: flex-end;
      gap: 12px;
      height: 220px;
      padding: 24px 10px 10px 10px;
      border-bottom: 2px solid var(--border);
      overflow-x: auto;
    }}
    .bar-col {{
      flex: 1;
      min-width: 52px;
      display: flex;
      flex-direction: column;
      align-items: center;
      height: 100%;
      justify-content: flex-end;
      cursor: pointer;
      position: relative;
    }}
    .bar-col:hover .bar-fill {{
      filter: brightness(0.92);
      transform: scaleY(1.02);
    }}
    .bar-col.active .bar-fill {{
      outline: 3px solid var(--primary);
      outline-offset: 2px;
    }}
    .bar-fill {{
      width: 100%;
      border-radius: 6px 6px 0 0;
      transition: all 0.2s ease;
      transform-origin: bottom;
      display: flex;
      align-items: flex-start;
      justify-content: center;
      padding-top: 6px;
      color: #ffffff;
      font-size: 12px;
      font-weight: 800;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.3);
    }}
    .bar-label {{
      font-size: 11px;
      font-weight: 700;
      color: var(--text-muted);
      margin-top: 8px;
      text-align: center;
      line-height: 1.25;
    }}

    /* Interactive Confusion Matrix (RQ3) */
    .matrix-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      margin-top: 10px;
    }}
    .matrix-table th, .matrix-table td {{
      padding: 10px 12px;
      text-align: center;
      border: 1px solid var(--border);
    }}
    .matrix-table th {{
      background: #f8fafc;
      font-weight: 700;
      color: var(--text);
    }}
    .matrix-table th.row-hdr {{
      text-align: left;
      font-weight: 600;
      background: #f1f5f9;
      color: var(--text);
    }}
    .matrix-cell {{
      font-weight: 700;
      cursor: pointer;
      transition: all 0.2s ease;
      border-radius: 4px;
    }}
    .matrix-cell:hover {{
      transform: scale(1.05);
      box-shadow: 0 0 0 2px var(--primary);
      z-index: 10;
    }}
    .matrix-cell.zero {{
      color: #94a3b8;
      background: #f8fafc;
      cursor: pointer;
    }}
    .matrix-cell.zero:hover {{
      transform: scale(1.03);
      box-shadow: 0 0 0 2px #cbd5e1;
    }}
    .matrix-cell.active {{
      outline: 3px solid var(--accent-rose);
      outline-offset: 1px;
      font-weight: 900;
      z-index: 20;
    }}

    /* Hotspot Inspector Card (RQ3) */
    .inspector-card {{
      background: #ffffff;
      border: 1px solid var(--border-dark);
      border-radius: 12px;
      padding: 16px 20px;
      margin-top: 16px;
      box-shadow: var(--shadow-sm);
    }}
    .inspector-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 12px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 10px;
    }}
    .inspector-title {{
      font-size: 14px;
      font-weight: 800;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .inspector-desc {{
      font-size: 12px;
      color: var(--text-muted);
      line-height: 1.5;
      margin-bottom: 14px;
      background: #f8fafc;
      padding: 10px 14px;
      border-left: 3px solid var(--primary);
      border-radius: 6px;
    }}
    .inspector-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
      gap: 12px;
      margin-bottom: 14px;
    }}
    .inspector-thumb-card {{
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
      cursor: pointer;
      transition: all 0.2s ease;
    }}
    .inspector-thumb-card:hover {{
      transform: translateY(-2px);
      box-shadow: var(--shadow-md);
      border-color: var(--primary);
    }}
    .inspector-thumb-card img {{
      width: 100%;
      height: 90px;
      object-fit: cover;
      display: block;
      background: #000;
    }}
    .inspector-thumb-meta {{
      padding: 6px 8px;
      font-size: 10px;
      color: var(--text-muted);
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: #ffffff;
    }}



    /* Active Filter Banner */
    .filter-banner {{
      display: none;
      align-items: center;
      justify-content: space-between;
      background: #eff6ff;
      border: 1px solid #bfdbfe;
      border-radius: 10px;
      padding: 10px 16px;
      margin-bottom: 20px;
      font-size: 13px;
      color: #1e40af;
      font-weight: 600;
    }}
    .filter-banner.active {{
      display: flex;
    }}
    .reset-btn {{
      background: #ffffff;
      border: 1px solid #bfdbfe;
      color: #1e40af;
      padding: 4px 12px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 700;
      cursor: pointer;
    }}
    .reset-btn:hover {{
      background: #dbeafe;
    }}



    /* Model Selector Tabs */
    .model-nav {{
      display: flex;
      background: #e2e8f0;
      border-radius: 10px;
      padding: 4px;
      gap: 4px;
      margin-bottom: 20px;
      width: fit-content;
    }}
    .model-btn {{
      background: transparent;
      border: none;
      color: var(--text-muted);
      padding: 8px 18px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
    }}
    .model-btn.active {{
      background: #ffffff;
      color: var(--primary);
      box-shadow: var(--shadow-sm);
    }}
    .model-btn:hover:not(.active) {{
      color: var(--text);
    }}

    /* Toolbar */
    .toolbar {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin-bottom: 24px;
      box-shadow: var(--shadow-sm);
    }}
    .control-group {{
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    label {{
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
    }}
    select, input[type="text"] {{
      background: #ffffff;
      border: 1px solid var(--border-dark);
      color: var(--text);
      padding: 8px 12px;
      border-radius: 8px;
      font-size: 13px;
      outline: none;
    }}
    select:focus {{
      border-color: var(--primary);
      box-shadow: 0 0 0 2px rgba(37,99,235,0.2);
    }}
    .slider-wrap {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    input[type="range"] {{
      flex: 1;
      accent-color: var(--primary);
    }}
    .slider-val {{
      font-size: 12px;
      font-family: monospace;
      color: var(--primary);
      font-weight: 700;
      width: 40px;
    }}

    /* Gallery Grid */
    .gallery {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(270px, 1fr));
      gap: 18px;
    }}
    .card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      transition: all 0.2s ease;
      cursor: pointer;
      box-shadow: var(--shadow-sm);
    }}
    .card:hover {{
      transform: translateY(-4px);
      box-shadow: var(--shadow-md);
      border-color: var(--primary);
    }}
    .card-img-wrap {{
      position: relative;
      width: 100%;
      height: 195px;
      padding: 7px;
      box-sizing: border-box;
      transition: all 0.2s ease;
    }}
    .card-img-inner {{
      position: relative;
      width: 100%;
      height: 100%;
      border-radius: 8px;
      overflow: hidden;
      background: #000000;
    }}
    .card-img-inner img {{
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }}
    .badge {{
      position: absolute;
      top: 8px;
      right: 8px;
      padding: 3px 8px;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 700;
    }}
    .badge-bg {{
      position: absolute;
      bottom: 8px;
      left: 8px;
      padding: 3px 8px;
      border-radius: 6px;
      font-size: 10px;
      font-weight: 600;
      background: rgba(255,255,255,0.92);
      color: var(--text);
      box-shadow: var(--shadow-sm);
      border: 1px solid var(--border);
    }}
    .badge-high {{ background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }}
    .badge-med {{ background: #fef3c7; color: #b45309; border: 1px solid #fcd34d; }}
    .badge-low {{ background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; }}

    .card-body {{
      padding: 12px;
      display: flex;
      flex-direction: column;
      flex: 1;
      justify-content: space-between;
    }}
    .card-species-header {{
      padding: 8px 10px;
      border-radius: 8px;
      margin-bottom: 8px;
      transition: all 0.2s ease;
    }}
    .card-species {{
      font-size: 14px;
      font-weight: 800;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .card-conf-expl {{
      font-size: 11px;
      color: var(--text-muted);
      margin-top: 3px;
      font-weight: 500;
      line-height: 1.35;
    }}
    .card-camera-box {{
      border-radius: 8px;
      padding: 9px 11px;
      margin-top: 10px;
      transition: all 0.2s ease;
    }}
    .card-meta {{
      font-size: 11px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .card-tags {{
      display: flex;
      gap: 6px;
      margin-top: 6px;
      flex-wrap: wrap;
    }}
    .tag {{
      background: #f1f5f9;
      padding: 2px 7px;
      border-radius: 4px;
      font-size: 10px;
      color: var(--text-muted);
      font-weight: 600;
    }}

    /* Empty state */
    .empty-state {{
      grid-column: 1 / -1;
      text-align: center;
      padding: 60px 20px;
      background: var(--card-bg);
      border: 1px dashed var(--border-dark);
      border-radius: 12px;
      color: var(--text-muted);
    }}

    /* Page Size Selector Pills */
    .page-size-pills {{
      display: inline-flex;
      background: #f1f5f9;
      border-radius: 8px;
      padding: 2px;
      gap: 2px;
    }}
    .page-size-btn {{
      background: transparent;
      border: none;
      color: var(--text-muted);
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 700;
      cursor: pointer;
      transition: all 0.2s ease;
    }}
    .page-size-btn:hover:not(.active) {{
      background: #e2e8f0;
      color: var(--text);
    }}
    .page-size-btn.active {{
      background: #ffffff;
      color: var(--primary);
      box-shadow: var(--shadow-sm);
    }}

    /* Pagination Bar */
    .pagination-btn {{
      background: #ffffff;
      border: 1px solid var(--border);
      color: var(--text);
      padding: 7px 14px;
      border-radius: 8px;
      font-size: 12px;
      font-weight: 700;
      cursor: pointer;
      transition: all 0.2s ease;
      box-shadow: var(--shadow-sm);
    }}
    .pagination-btn:hover:not(:disabled) {{
      border-color: var(--primary);
      color: var(--primary);
      background: var(--primary-light);
    }}
    .pagination-btn:disabled {{
      opacity: 0.4;
      cursor: not-allowed;
      border-color: var(--border);
    }}
    .pagination-btn.active {{
      background: var(--primary);
      color: #ffffff;
      border-color: var(--primary);
    }}
    .pagination-info {{
      font-size: 12px;
      font-weight: 600;
      color: var(--text-muted);
      margin: 0 8px;
    }}

    /* Modal */
    .modal-backdrop {{
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(15, 23, 42, 0.7);
      backdrop-filter: blur(4px);
      z-index: 1000;
      align-items: center;
      justify-content: center;
      padding: 20px;
    }}
    .modal-backdrop.active {{
      display: flex;
    }}
    .modal {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 16px;
      max-width: 860px;
      width: 100%;
      max-height: 94vh;
      overflow-y: auto;
      padding: 24px;
      position: relative;
      box-shadow: var(--shadow-lg);
    }}
    .modal-close {{
      position: absolute;
      top: 16px;
      right: 16px;
      background: #f1f5f9;
      border: 1px solid var(--border);
      color: var(--text);
      font-size: 20px;
      width: 32px;
      height: 32px;
      border-radius: 50%;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    .modal-close:hover {{
      background: #e2e8f0;
    }}
    .modal-img {{
      width: 100%;
      max-height: 440px;
      object-fit: contain;
      background: #000;
      border-radius: 8px;
      margin-top: 12px;
    }}
    .modal-nav-btns {{
      display: flex;
      justify-content: space-between;
      margin-top: 12px;
    }}
    .nav-btn {{
      background: #f1f5f9;
      border: 1px solid var(--border-dark);
      color: var(--text);
      padding: 7px 16px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 700;
      cursor: pointer;
    }}
    .nav-btn:hover {{
      background: #e2e8f0;
    }}
    .top5-table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 14px;
      font-size: 12px;
    }}
    .top5-table th, .top5-table td {{
      padding: 8px 10px;
      text-align: left;
      border-bottom: 1px solid var(--border);
    }}
    .top5-table th {{
      background: #f8fafc;
      font-weight: 700;
      color: var(--text-muted);
    }}
    .prob-bar {{
      height: 8px;
      border-radius: 4px;
      background: var(--primary);
    }}
  </style>
</head>
<body>

  <!-- Header -->
  <header>
    <div>
      <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
        <span style="font-size: 26px;">🌿</span>
        <h1 style="margin: 0; font-size: 24px; font-weight: 800; letter-spacing: -0.02em;">Phantom Wildlife Explorer</h1>
      </div>
      <p class="sub">Why AI Sees Animals in Empty Forests: An interactive audit of environmental illusions, camera hotspots, and decision certainty in wildlife monitoring.</p>
    </div>
    <div class="stat-pills">
      <div class="pill">
        <span class="pill-label">Total Empty Photos</span>
        <span class="pill-val">2,399 Photos</span>
        <span class="pill-sub">Tested validation scenes with no animals</span>
      </div>
      <div class="pill">
        <span class="pill-label">AI False Alarms</span>
        <span class="pill-val" style="color: var(--accent-rose);">118 Hallucinations</span>
        <span class="pill-sub">4.92% of empty scenes tricked the model</span>
      </div>
      <div class="pill">
        <span class="pill-label">Top Phantom Species</span>
        <span class="pill-val" style="color: #e11d48;">Panthera onca & Leopardus pardalis (27%)</span>
        <span class="pill-sub">32 empty frames mistaken for Panthera onca & Leopardus pardalis</span>
      </div>
      <div class="pill">
        <span class="pill-label">Model Improvement</span>
        <span class="pill-val" style="color: var(--accent-green);">48% Fewer Errors</span>
        <span class="pill-sub">Fine-tuned model cut errors from 225 down to 118</span>
      </div>
      <div class="pill">
        <span class="pill-label">Problem Locations</span>
        <span class="pill-val" style="color: var(--primary);">3 Cameras = 54% Errors</span>
        <span class="pill-sub">Over half of all mistakes happen at just 3 sites</span>
      </div>
    </div>
  </header>

  <!-- Educational Explainer Banner: What is an AI Hallucination? -->
  <div class="explainer-card">
    <div class="explainer-header">
      <div class="explainer-title">
        <span style="font-size: 20px;">💡</span>
        <span>What is an AI "Hallucination" in Wildlife Camera Traps?</span>
      </div>
    </div>
    <div class="explainer-body">
      <p>
        An <strong>AI Hallucination</strong> (false alarm) occurs when the computer vision model looks at an entirely empty photo and confidently claims it sees an animal. This isn't random guessing: the AI gets fooled by environmental textures in the forest — such as camera flash reflections mimicking <em>Panthera onca</em> spots, or curved tree branches resembling the limbs of <em>Equus ferus</em>.
      </p>
      <div class="explainer-features">
        <div class="feature-item feature-blue active" onclick="openTabAndScroll('chart')">
          <div>
            <strong style="color: #2563eb;">📍 Problem Cameras:</strong> Over <strong>54% of all errors</strong> occur at just 3 specific camera sites with tricky backgrounds.
          </div>
          <div class="feature-link" style="color: #2563eb;">
            <span>Explore Camera Hotspots &rarr;</span>
          </div>
        </div>
        <div class="feature-item feature-rose" onclick="openTabAndScroll('species')">
          <div>
            <strong style="color: #e11d48;">🐾 Phantom Species:</strong> Over <strong>27% of false alarms</strong> are spotted wildcats (<em>Panthera onca & Leopardus pardalis</em>), triggered by flash glints.
          </div>
          <div class="feature-link" style="color: #e11d48;">
            <span>Explore Hallucinated Animals &rarr;</span>
          </div>
        </div>
        <div class="feature-item feature-green" onclick="openTabAndScroll('matrix')">
          <div>
            <strong style="color: #059669;">🌿 Environment Triggers:</strong> <strong>61.9% of false alarms</strong> happen at night, where flash glare on foliage is mistaken for wildcats.
          </div>
          <div class="feature-link" style="color: #059669;">
            <span>Inspect Environment Matrix &rarr;</span>
          </div>
        </div>
        <div class="feature-item feature-purple" onclick="openTabAndScroll('burst')">
          <div>
            <strong style="color: #7c3aed;">⏱️ Photo Bursts:</strong> In <strong>48.6% of multi-shot bursts</strong>, the AI repeats the exact same mistake across all 10 consecutive photos!
          </div>
          <div class="feature-link" style="color: #7c3aed;">
            <span>Analyze Burst Dynamics &rarr;</span>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Tab 1: Interactive Camera Bar Chart -->
  <div class="tab-pane active" id="pane-chart">
    <div class="pane-header">
      <div>
        <div class="pane-title" style="color: #1e3a8a;">
          <span>📍</span>
          <span>Camera Hotspots: Which camera locations trigger the most false alarms?</span>
          <span class="pane-badge" style="background: #eff6ff; color: #1e40af; border-color: #bfdbfe;">Click Any Bar to Filter Gallery</span>
        </div>
        <div class="pane-desc" id="camera-chart-desc">
          Click any bar to instantly filter the photo gallery below to that camera's false detection frames.
        </div>
      </div>
      <div style="display: flex; gap: 8px; align-items: center;">
        <span style="font-size: 11px; font-weight: 700; color: var(--text-dim); text-transform: uppercase;">Bar Metric:</span>
        <button class="nav-btn active" id="btn-chart-count" onclick="setChartMode('count')" style="background: var(--primary); color: #ffffff;">Count of False Alarms</button>
        <button class="nav-btn" id="btn-chart-rate" onclick="setChartMode('rate')" style="background: #f1f5f9; color: var(--text);">Error Rate (% of Empty Frames)</button>
      </div>
    </div>
    <div class="chart-wrap" id="camera-barchart">
      <!-- Bars dynamically generated in JS -->
    </div>
  </div>

  <!-- Tab: Phantom Species Statistics -->
  <div class="tab-pane" id="pane-species">
    <div class="pane-header">
      <div>
        <div class="pane-title" style="color: #be123c;">
          <span>🐾</span>
          <span>Phantom Species: Which animals does the AI hallucinate most often?</span>
          <span class="pane-badge" style="background: #fff1f2; color: #be123c; border-color: #fecdd3;">Click Any Animal to Filter Gallery Below</span>
        </div>
        <div class="pane-desc">
          Auditing which wildlife species the model falsely detects in completely empty photos. Click any species row to immediately inspect all matching false detection images.
        </div>
      </div>
      <div style="display: flex; gap: 8px; align-items: center;">
        <span style="font-size: 11px; font-weight: 700; color: var(--text-dim); text-transform: uppercase;">Sort By:</span>
        <button class="nav-btn active" id="btn-species-count" onclick="setSpeciesSort('count')" style="background: #e11d48; color: #ffffff;">Count of False Alarms</button>
        <button class="nav-btn" id="btn-species-conf" onclick="setSpeciesSort('conf')" style="background: #f1f5f9; color: var(--text);">Average Model Certainty</button>
      </div>
    </div>

    <!-- 3 Key Hallucination Clusters Callout Cards -->
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; margin-bottom: 16px;">
      <div style="background: #fff1f2; border: 1px solid #fecdd3; border-radius: 10px; padding: 12px 14px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <span style="font-size: 11px; font-weight: 700; color: #be123c; text-transform: uppercase;">🐆 Panthera onca & Leopardus pardalis</span>
          <span style="font-size: 10px; font-weight: 700; background: #ffe4e6; color: #9f1239; padding: 2px 6px; border-radius: 4px;">27.1% (32 Frames)</span>
        </div>
        <div style="font-size: 15px; font-weight: 800; color: #9f1239; margin-top: 4px;">Panthera onca (18) & Leopardus pardalis (14)</div>
        <div style="font-size: 11px; color: #be123c; margin-top: 3px; line-height: 1.4;">
          <strong>96.9% happen at Night.</strong> Monochrome IR flash glints on waxy leaves mimic circular coat rosettes.
        </div>
      </div>

      <div style="background: #f5f3ff; border: 1px solid #ddd6fe; border-radius: 10px; padding: 12px 14px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <span style="font-size: 11px; font-weight: 700; color: #6d28d9; text-transform: uppercase;">🐎 Equus ferus</span>
          <span style="font-size: 10px; font-weight: 700; background: #ede9fe; color: #5b21b6; padding: 2px 6px; border-radius: 4px;">Highest Conviction (87%)</span>
        </div>
        <div style="font-size: 15px; font-weight: 800; color: #5b21b6; margin-top: 4px;">Equus ferus (10 Frames)</div>
        <div style="font-size: 11px; color: #6d28d9; margin-top: 3px; line-height: 1.4;">
          <strong>100% in Dense Foliage.</strong> Camera 456's vertical tree branches and curved logs trick the model into seeing legs (87.1% avg conf).
        </div>
      </div>

      <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 10px; padding: 12px 14px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <span style="font-size: 11px; font-weight: 700; color: #166534; text-transform: uppercase;">🦃 Crax rubra, Dasyprocta & Meleagris</span>
          <span style="font-size: 10px; font-weight: 700; background: #dcfce7; color: #14532d; padding: 2px 6px; border-radius: 4px;">22.9% (27 Frames)</span>
        </div>
        <div style="font-size: 15px; font-weight: 800; color: #14532d; margin-top: 4px;">Crax rubra (10), Dasyprocta punctata (9), Meleagris ocellata (8)</div>
        <div style="font-size: 11px; color: #15803d; margin-top: 3px; line-height: 1.4;">
          <strong>Forest Floor Illusions.</strong> Leaf mounds, root knots, and ground shadows mimic foraging ground wildlife.
        </div>
      </div>
    </div>

    <!-- Ranked Species List -->
    <div style="background: #ffffff; border: 1px solid var(--border); border-radius: 12px; overflow-x: auto; box-shadow: var(--shadow-sm);">
      <div id="species-ranked-container" style="min-width: 680px;">
        <!-- Generated in JS -->
      </div>
    </div>
  </div>

  <!-- Tab 2: Environment Triggers Matrix -->
  <div class="tab-pane" id="pane-matrix">
    <div class="pane-header">
      <div>
        <div class="pane-title" style="color: #065f46;">
          <span>🌿</span>
          <span>Environment Triggers: What natural backgrounds fool the model?</span>
          <span class="pane-badge" style="background: #ecfdf5; color: #065f46; border-color: #a7f3d0;">Click Any Cell to Inspect Hotspots & Images</span>
        </div>
        <div class="pane-desc">
          Cross-table showing how specific lighting conditions and foliage patterns trigger phantom animal sightings. Click any cell to view diagnostic insights, camera hotspots, and example frames below.
        </div>
      </div>
      <div style="display: flex; gap: 8px; align-items: center;">
        <span style="font-size: 11px; font-weight: 700; color: var(--text-dim); text-transform: uppercase;">Matrix Metric:</span>
        <button class="nav-btn active" id="btn-matrix-count" onclick="setMatrixMode('count')" style="background: var(--primary); color: #ffffff;">Count of False Alarms</button>
        <button class="nav-btn" id="btn-matrix-pct" onclick="setMatrixMode('pct')" style="background: #f1f5f9; color: var(--text);">% of Background Errors (Row %)</button>
      </div>
    </div>

    <!-- 3 Key Insight Callouts -->
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; margin-bottom: 16px;">
      <div style="background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 10px; padding: 12px 14px;">
        <div style="font-size: 11px; font-weight: 700; color: #1e40af; text-transform: uppercase;">🌙 Night / IR Flash (73 FPs, 61.9%)</div>
        <div style="font-size: 16px; font-weight: 800; color: #1e3a8a; margin-top: 2px;">42.5% Panthera onca & Leopardus pardalis (31 Frames)</div>
        <div style="font-size: 11px; color: #3b82f6; margin-top: 3px;">Monochrome flash reflections on foliage mimic circular coat rosettes.</div>
      </div>
      <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 10px; padding: 12px 14px;">
        <div style="font-size: 11px; font-weight: 700; color: #166534; text-transform: uppercase;">🌿 Dense Foliage (36 FPs, 30.5%)</div>
        <div style="font-size: 16px; font-weight: 800; color: #14532d; margin-top: 2px;">83.3% Forest Species (30 Frames)</div>
        <div style="font-size: 11px; color: #15803d; margin-top: 3px;">Branches & leaf mounds mimic Crax rubra, Equus ferus & Urocyon cinereoargenteus.</div>
      </div>
      <div style="background: #fefce8; border: 1px solid #fef08a; border-radius: 10px; padding: 12px 14px;">
        <div style="font-size: 11px; font-weight: 700; color: #854d0e; text-transform: uppercase;">☀️ Sunlit & Open Landscapes (9 FPs, 7.6%)</div>
        <div style="font-size: 16px; font-weight: 800; color: #713f12; margin-top: 2px;">100% Savanna Species (9 Frames)</div>
        <div style="font-size: 11px; color: #a16207; margin-top: 3px;">Open terrain features trigger strong priors for Aepyceros melampus, Crocuta crocuta & Bos taurus.</div>
      </div>
    </div>

    <div style="overflow-x: auto; background: #ffffff; border: 1px solid var(--border); border-radius: 10px; padding: 6px;">
      <table class="matrix-table" id="confusion-matrix-table">
        <!-- Matrix dynamically generated in JS -->
      </table>
    </div>

    <!-- Interactive Hotspot Inspector Card -->
    <div id="matrix-inspector">
      <!-- Generated in JS when cell is selected -->
    </div>
  </div>



  <!-- Tab 4: Rapid Photo Bursts -->
  <div class="tab-pane" id="pane-burst">
    <div class="pane-header">
      <div>
        <div class="pane-title" style="color: #5b21b6;">
          <span>⏱️</span>
          <span>Rapid Photo Bursts: Are false alarms single glitches or persistent illusions?</span>
          <span class="pane-badge" style="background: #f5f3ff; color: #5b21b6; border-color: #ddd6fe;">48.6% Persistent Illusions</span>
        </div>
        <div class="pane-desc">
          When motion triggers a camera trap, it captures a rapid burst of 3 to 10 photos in just a few seconds. We audit whether the AI makes an isolated 1-shot mistake, or gets continuously fooled across the entire burst.
        </div>
      </div>
    </div>

    <!-- 3 Intuitive Category Cards -->
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin-top: 10px;">
      <div style="background: #f5f3ff; border: 1px solid #ddd6fe; border-radius: 12px; padding: 16px;">
        <div style="font-size: 11px; font-weight: 700; color: #6d28d9; text-transform: uppercase;">100% Persistent Illusions</div>
        <div style="font-size: 26px; font-weight: 800; color: #5b21b6; margin-top: 4px;">48.6% (17 Bursts)</div>
        <div style="font-size: 12px; color: #6d28d9; margin-top: 4px; line-height: 1.4;">Every single photo in the multi-shot burst was falsely classified as an animal!</div>
      </div>
      <div style="background: #fffbeb; border: 1px solid #fde68a; border-radius: 12px; padding: 16px;">
        <div style="font-size: 11px; font-weight: 700; color: #b45309; text-transform: uppercase;">Partial Multi-Shot Bursts</div>
        <div style="font-size: 26px; font-weight: 800; color: #92400e; margin-top: 4px;">25.7% (9 Bursts)</div>
        <div style="font-size: 12px; color: #b45309; margin-top: 4px; line-height: 1.4;">The hallucination persisted across 2 to 6 consecutive photos before disappearing.</div>
      </div>
      <div style="background: #f8fafc; border: 1px solid var(--border); border-radius: 12px; padding: 16px;">
        <div style="font-size: 11px; font-weight: 700; color: var(--text-dim); text-transform: uppercase;">Isolated 1-Shot Glitches</div>
        <div style="font-size: 26px; font-weight: 800; color: var(--text); margin-top: 4px;">25.7% (9 Bursts)</div>
        <div style="font-size: 12px; color: var(--text-muted); margin-top: 4px; line-height: 1.4;">A single momentary mistake; all other photos in the burst were correctly labeled empty.</div>
      </div>
    </div>

    <!-- Case Study Spotlight Box -->
    <div style="margin-top: 16px; background: #ffffff; border: 1px solid #ddd6fe; border-radius: 12px; padding: 18px 20px;">
      <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 10px;">
        <div style="font-weight: 800; font-size: 14px; color: #5b21b6; display: flex; align-items: center; gap: 8px;">
          <span>🎯 Spotlight Case Study:</span>
          <span>Camera 456 (Forest Understory)</span>
        </div>
        <span style="font-size: 11px; font-weight: 700; background: #f5f3ff; color: #6d28d9; padding: 3px 10px; border-radius: 9999px;">
          10 Photos in 8.2s &bull; 100% Equus ferus
        </span>
      </div>
      <p style="font-size: 12px; color: var(--text-muted); line-height: 1.5; margin-bottom: 12px;">
        In this 10-shot burst sequence, the camera triggered repeatedly over 8.2 seconds on a completely empty forest clearing. The AI predicted <strong>Equus ferus across all 10 consecutive photos</strong> with confidence scores climbing as high as <strong>98.7%</strong>. Vertical tree trunks and hanging lianas in the camera's fixed view permanently matched the leg and body contours of <em>Equus ferus</em>!
      </p>
      
      <!-- Visual Timeline Filmstrip Blocks -->
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(85px, 1fr)); gap: 6px;">
        <div style="background: #f5f3ff; border: 1px solid #c4b5fd; border-radius: 6px; padding: 8px 4px; text-align: center;">
          <div style="font-size: 10px; font-weight: 700; color: #6d28d9;">Frame 1</div>
          <div style="font-size: 9px; color: #7c3aed; margin-top: 2px;">+0.0s</div>
          <div style="font-size: 10px; font-weight: 800; color: #dc2626; margin-top: 4px;">Equus ferus 74%</div>
        </div>
        <div style="background: #f5f3ff; border: 1px solid #c4b5fd; border-radius: 6px; padding: 8px 4px; text-align: center;">
          <div style="font-size: 10px; font-weight: 700; color: #6d28d9;">Frame 2</div>
          <div style="font-size: 9px; color: #7c3aed; margin-top: 2px;">+1.8s</div>
          <div style="font-size: 10px; font-weight: 800; color: #dc2626; margin-top: 4px;">Equus ferus 94%</div>
        </div>
        <div style="background: #f5f3ff; border: 1px solid #c4b5fd; border-radius: 6px; padding: 8px 4px; text-align: center;">
          <div style="font-size: 10px; font-weight: 700; color: #6d28d9;">Frame 3</div>
          <div style="font-size: 9px; color: #7c3aed; margin-top: 2px;">+3.6s</div>
          <div style="font-size: 10px; font-weight: 800; color: #dc2626; margin-top: 4px;">Equus ferus 97%</div>
        </div>
        <div style="background: #f5f3ff; border: 1px solid #c4b5fd; border-radius: 6px; padding: 8px 4px; text-align: center;">
          <div style="font-size: 10px; font-weight: 700; color: #6d28d9;">Frame 4</div>
          <div style="font-size: 9px; color: #7c3aed; margin-top: 2px;">+4.1s</div>
          <div style="font-size: 10px; font-weight: 800; color: #dc2626; margin-top: 4px;">Equus ferus 64%</div>
        </div>
        <div style="background: #f5f3ff; border: 1px solid #c4b5fd; border-radius: 6px; padding: 8px 4px; text-align: center;">
          <div style="font-size: 10px; font-weight: 700; color: #6d28d9;">Frame 5</div>
          <div style="font-size: 9px; color: #7c3aed; margin-top: 2px;">+4.9s</div>
          <div style="font-size: 10px; font-weight: 800; color: #dc2626; margin-top: 4px;">Equus ferus 82%</div>
        </div>
        <div style="background: #f5f3ff; border: 1px solid #c4b5fd; border-radius: 6px; padding: 8px 4px; text-align: center;">
          <div style="font-size: 10px; font-weight: 700; color: #6d28d9;">Frame 6</div>
          <div style="font-size: 9px; color: #7c3aed; margin-top: 2px;">+5.3s</div>
          <div style="font-size: 10px; font-weight: 800; color: #dc2626; margin-top: 4px;">Equus ferus 68%</div>
        </div>
        <div style="background: #f5f3ff; border: 1px solid #c4b5fd; border-radius: 6px; padding: 8px 4px; text-align: center;">
          <div style="font-size: 10px; font-weight: 700; color: #6d28d9;">Frame 7</div>
          <div style="font-size: 9px; color: #7c3aed; margin-top: 2px;">+6.0s</div>
          <div style="font-size: 10px; font-weight: 800; color: #dc2626; margin-top: 4px;">Equus ferus 97%</div>
        </div>
        <div style="background: #f5f3ff; border: 1px solid #c4b5fd; border-radius: 6px; padding: 8px 4px; text-align: center;">
          <div style="font-size: 10px; font-weight: 700; color: #6d28d9;">Frame 8</div>
          <div style="font-size: 9px; color: #7c3aed; margin-top: 2px;">+6.9s</div>
          <div style="font-size: 10px; font-weight: 800; color: #dc2626; margin-top: 4px;">Equus ferus 98%</div>
        </div>
        <div style="background: #f5f3ff; border: 1px solid #c4b5fd; border-radius: 6px; padding: 8px 4px; text-align: center;">
          <div style="font-size: 10px; font-weight: 700; color: #6d28d9;">Frame 9</div>
          <div style="font-size: 9px; color: #7c3aed; margin-top: 2px;">+7.6s</div>
          <div style="font-size: 10px; font-weight: 800; color: #dc2626; margin-top: 4px;">Equus ferus 99%</div>
        </div>
        <div style="background: #f5f3ff; border: 1px solid #c4b5fd; border-radius: 6px; padding: 8px 4px; text-align: center;">
          <div style="font-size: 10px; font-weight: 700; color: #6d28d9;">Frame 10</div>
          <div style="font-size: 9px; color: #7c3aed; margin-top: 2px;">+8.2s</div>
          <div style="font-size: 10px; font-weight: 800; color: #dc2626; margin-top: 4px;">Equus ferus 99%</div>
        </div>
      </div>
    </div>
  </div>

  <!-- Active Filter Notification Banner -->
  <div class="filter-banner" id="active-filter-banner">
    <span id="active-filter-text">Active Filter: Showing all cameras</span>
    <button class="reset-btn" onclick="resetAllFilters()">✕ Clear Filter / Show All</button>
  </div>



  <!-- Model Selector Tabs -->
  <div class="model-nav">
    <button class="model-btn active" onclick="switchModel('fine_tuned')">WILDS Fine-Tuned ResNet-50 (118 FPs)</button>
    <button class="model-btn" onclick="switchModel('probe')">ImageNet Linear Probe (225 FPs)</button>
  </div>

  <!-- Filter Toolbar -->
  <div class="toolbar" id="gallery-section">
    <div class="control-group">
      <label>Filter by Species</label>
      <select id="species-filter" onchange="filterGallery()">
        <option value="">All Predicted Species</option>
        {species_options}
      </select>
    </div>

    <div class="control-group">
      <label>Visual Background</label>
      <select id="bg-filter" onchange="filterGallery()">
        <option value="">All Background Types</option>
        {bg_options}
      </select>
    </div>

    <div class="control-group">
      <label>Camera Location</label>
      <select id="cam-filter" onchange="filterGallery()">
        <option value="">All Camera Locations</option>
        {camera_options}
      </select>
    </div>

    <div class="control-group">
      <label>Min Model Confidence</label>
      <div class="slider-wrap">
        <input type="range" id="conf-slider" min="0" max="100" value="0" oninput="updateConf(this.value)">
        <span class="slider-val" id="conf-val">0%</span>
      </div>
    </div>

    <div class="control-group">
      <label>Sort By</label>
      <select id="sort-select" onchange="filterGallery()">
        <option value="conf_desc">Confidence (High to Low)</option>
        <option value="conf_asc">Confidence (Low to High)</option>
        <option value="cam">Camera Location</option>
        <option value="species">Predicted Species</option>
        <option value="lum_desc">Darkness / Luminance</option>
        <option value="green_desc">Vegetation Density</option>
      </select>
    </div>
  </div>

  <div style="background: #ffffff; border: 1px solid var(--border); border-radius: 12px; padding: 12px 16px; margin-bottom: 18px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; box-shadow: var(--shadow-sm);">
    <div style="display: flex; align-items: center; gap: 14px; flex-wrap: wrap;">
      <span style="font-size: 13px; font-weight: 700; color: var(--text);">
        Displaying <span id="stat-displayed" style="color: var(--primary); font-weight: 800;">0</span> of <span id="stat-count" style="color: var(--text); font-weight: 800;">0</span> false detection frames:
      </span>
      <div style="display: flex; align-items: center; gap: 6px;">
        <span style="font-size: 11px; font-weight: 700; color: var(--text-dim); text-transform: uppercase;">Show:</span>
        <div class="page-size-pills" id="page-size-selector">
          <button class="page-size-btn" onclick="setPageSize(12)">12</button>
          <button class="page-size-btn active" onclick="setPageSize(24)">24</button>
          <button class="page-size-btn" onclick="setPageSize(36)">36</button>
          <button class="page-size-btn" onclick="setPageSize(48)">48</button>
          <button class="page-size-btn" onclick="setPageSize('all')">All</button>
        </div>
      </div>
    </div>

    <!-- Visual Legend Pills -->
    <div style="font-size: 11px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
      <span style="background: #fef3c7; border: 1px solid #fde68a; padding: 3px 8px; border-radius: 6px; color: #92400e; font-weight: 700;">🐾 Background behind image = Species</span>
      <span style="background: #eff6ff; border: 1px solid #bfdbfe; padding: 3px 8px; border-radius: 6px; color: #1e40af; font-weight: 700;">📷 Bottom box = Camera Location</span>
    </div>
  </div>

  <!-- Gallery Grid -->
  <div class="gallery" id="gallery-container"></div>

  <!-- Pagination Controls -->
  <div id="gallery-pagination" style="display: flex; justify-content: center; align-items: center; gap: 8px; margin-top: 24px; margin-bottom: 24px; flex-wrap: wrap;"></div>

  <!-- Inspection Modal -->
  <div class="modal-backdrop" id="modal" onclick="closeModal(event)">
    <div class="modal" onclick="event.stopPropagation()">
      <button class="modal-close" onclick="closeModal()">&times;</button>
      
      <div class="modal-nav-btns">
        <button class="nav-btn" onclick="prevModalItem()">&larr; Previous Frame</button>
        <button class="nav-btn" onclick="nextModalItem()">Next Frame &rarr;</button>
      </div>

      <h2 id="modal-title" style="font-size: 22px; font-weight: 800; text-transform: capitalize; margin-top: 10px;">Species</h2>
      <p id="modal-sub" style="font-size: 12px; color: var(--text-muted);"></p>
      <img id="modal-img" class="modal-img" src="" alt="Camera trap image">
      
      <h3 style="font-size: 14px; font-weight: 700; margin-top: 18px; color: var(--text);">Top-5 Model Class Probability Distribution:</h3>
      <table class="top5-table" id="modal-top5"></table>

      <div style="margin-top: 18px; padding: 14px; background: #eff6ff; border-left: 3px solid var(--primary); font-size: 12px; border-radius: 6px; color: #1e40af;">
        <strong>Diagnostic Analysis:</strong> Notice whether the true class <code>empty</code> ranked within the top 5, or if the model's confidence was divided entirely across animal classes due to background vegetation/lighting textures.
      </div>
    </div>
  </div>

  <script>
    const dataStore = {data_json};
    let currentModel = 'fine_tuned';
    let currentFiltered = [];
    let currentModalIndex = 0;
    let activeCameraFilter = '';
    let activeMatrixCell = null;
    let currentPage = 1;
    let currentPageSize = 24;

    function formatScientific(sp) {{
      if (!sp) return '';
      const parts = sp.trim().replace(/_/g, ' ').split(/\\s+/);
      if (parts.length >= 2) {{
        return parts[0].charAt(0).toUpperCase() + parts[0].slice(1).toLowerCase() + ' ' + parts.slice(1).join(' ').toLowerCase();
      }}
      return parts[0].charAt(0).toUpperCase() + parts[0].slice(1).toLowerCase();
    }}

    function setPageSize(size) {{
      currentPageSize = size === 'all' ? 'all' : parseInt(size);
      currentPage = 1;

      document.querySelectorAll('.page-size-btn').forEach(btn => {{
        const isAll = size === 'all' && btn.innerText.trim().toLowerCase() === 'all';
        const isNum = parseInt(btn.innerText.trim()) === size;
        if (isAll || isNum) {{
          btn.classList.add('active');
        }} else {{
          btn.classList.remove('active');
        }}
      }});

      renderGalleryPage();
    }}

    function goToPage(page) {{
      currentPage = page;
      renderGalleryPage();
      document.getElementById('gallery-container').scrollIntoView({{ behavior: 'smooth', block: 'start' }});
    }}

    function prevPage() {{
      if (currentPage > 1) {{
        goToPage(currentPage - 1);
      }}
    }}

    function nextPage() {{
      const totalItems = currentFiltered.length;
      const pageSize = currentPageSize === 'all' ? totalItems : currentPageSize;
      const maxPage = Math.ceil(totalItems / pageSize) || 1;
      if (currentPage < maxPage) {{
        goToPage(currentPage + 1);
      }}
    }}

    // Switch analytical tabs
    function switchTab(tabId) {{
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
      document.querySelectorAll('.feature-item').forEach(f => f.classList.remove('active'));
      const targetFeature = document.querySelector(`.feature-item[onclick*="${{tabId}}"]`);
      if (targetFeature) targetFeature.classList.add('active');
      const pane = document.getElementById('pane-' + tabId);
      if (pane) pane.classList.add('active');
    }}

    // Switch tab and smooth scroll to it
    function openTabAndScroll(tabId) {{
      switchTab(tabId);
      const pane = document.getElementById('pane-' + tabId);
      if (pane) {{
        pane.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
      }}
    }}

    let currentChartMode = 'count';

    function setChartMode(mode) {{
      currentChartMode = mode;
      document.getElementById('btn-chart-count').style.background = mode === 'count' ? 'var(--primary)' : '#f1f5f9';
      document.getElementById('btn-chart-count').style.color = mode === 'count' ? '#ffffff' : 'var(--text)';
      document.getElementById('btn-chart-rate').style.background = mode === 'rate' ? 'var(--primary)' : '#f1f5f9';
      document.getElementById('btn-chart-rate').style.color = mode === 'rate' ? '#ffffff' : 'var(--text)';
      const desc = document.getElementById('camera-chart-desc');
      if (desc) {{
        if (mode === 'rate') {{
          desc.innerHTML = `Showing <b>Error Rate on Empty Frames</b>: What percentage of blank/animal-free triggers at each camera fooled the AI. (Note: These cameras take hundreds of photos with real animals; this metric tests how reliably the model rejects truly empty triggers).`;
        }} else {{
          desc.innerHTML = `Click any bar to instantly filter the photo gallery below to that camera's false detection frames.`;
        }}
      }}
      renderCameraBarChart(mode);
    }}

    // Render Camera Bar Chart
    function renderCameraBarChart(mode = 'count') {{
      let cameras = [...(dataStore.cameras || [])];
      if (mode === 'rate') {{
        cameras.sort((a, b) => (b.fpr_val - a.fpr_val) || (b.n_fp - a.n_fp));
      }} else {{
        cameras.sort((a, b) => (b.n_fp - a.n_fp) || (b.fpr_val - a.fpr_val));
      }}
      const chart = document.getElementById('camera-barchart');
      const maxCount = Math.max(...cameras.map(c => c.n_fp), 1);

      let html = '';
      cameras.forEach(c => {{
        let heightPct = 0;
        let displayVal = '';
        
        if (mode === 'count') {{
          heightPct = Math.max((c.n_fp / maxCount) * 100, 12);
          displayVal = c.n_fp; // e.g. 34, 20, 10
        }} else {{
          heightPct = Math.max(c.fpr_val * 100, 12);
          displayVal = c.fpr_pct; // e.g. 94.4%, 100%
        }}
        
        // Color by severity
        let barColor = '#3b82f6'; // royal blue
        if (c.n_fp >= 20 || (mode === 'rate' && c.fpr_val >= 0.8)) barColor = '#ef4444'; // rose red
        else if (c.n_fp >= 10 || (mode === 'rate' && c.fpr_val >= 0.2)) barColor = '#f59e0b'; // amber

        const isActive = String(c.camera) === String(activeCameraFilter) ? ' active' : '';
        const countSub = mode === 'rate' ? `${{c.n_fp}} / ${{c.n_empty}} empty` : `${{c.n_fp}} of ${{c.n_empty}}`;
        const totalPhotos = c.n_total ? c.n_total.toLocaleString() : 'many';
        const contextSub = mode === 'rate' && c.n_total ? `<div style="font-size: 8px; color: #64748b; margin-top: 1px;" title="${{c.n_fp}} errors out of ${{totalPhotos}} total photos at this site">(${{c.overall_fpr_pct}} of ${{totalPhotos}})</div>` : '';
        const tooltip = `Camera ${{c.camera}}: ${{c.n_fp}} false alarms out of ${{c.n_empty}} empty photos (${{c.fpr_pct}} error rate on empty triggers)\\nTotal photos captured at this site: ${{totalPhotos}} photos (${{c.overall_fpr_pct || 'N/A'}} overall false alarm rate)\\nPrimary hallucinated species: ${{formatScientific(c.top_species)}}`;

        html += `
          <div class="bar-col${{isActive}}" id="barchart-cam-${{c.camera}}" onclick="selectCamera('${{c.camera}}')" 
               title="${{tooltip}}">
            <div class="bar-fill" style="height: ${{heightPct}}%; background: ${{barColor}};">
              ${{displayVal}}
            </div>
            <div class="bar-label">
              <div>Cam ${{c.camera}}</div>
              <div style="font-size: 9px; color: var(--text-dim); font-weight: 600; margin-top: 2px;">${{countSub}}</div>
              ${{contextSub}}
            </div>
          </div>
        `;
      }});
      chart.innerHTML = html;
    }}

    let currentSpeciesSort = 'count';

    function setSpeciesSort(mode) {{
      currentSpeciesSort = mode;
      document.getElementById('btn-species-count').style.background = mode === 'count' ? '#e11d48' : '#f1f5f9';
      document.getElementById('btn-species-count').style.color = mode === 'count' ? '#ffffff' : 'var(--text)';
      document.getElementById('btn-species-conf').style.background = mode === 'conf' ? '#e11d48' : '#f1f5f9';
      document.getElementById('btn-species-conf').style.color = mode === 'conf' ? '#ffffff' : 'var(--text)';
      renderSpeciesStats();
    }}

    // Render Ranked Hallucinated Species List
    function renderSpeciesStats() {{
      const container = document.getElementById('species-ranked-container');
      if (!container) return;
      const sppData = dataStore.species || {{}};
      const sppList = [...(sppData[currentModel] || sppData['fine_tuned'] || [])];
      if (sppList.length === 0) {{
        container.innerHTML = '<div style="padding: 24px; text-align: center; color: var(--text-muted);">No species hallucination data recorded.</div>';
        return;
      }}

      if (currentSpeciesSort === 'conf') {{
        sppList.sort((a, b) => (b.mean_conf_val - a.mean_conf_val) || (b.count - a.count));
      }} else {{
        sppList.sort((a, b) => (b.count - a.count) || (b.mean_conf_val - a.mean_conf_val));
      }}

      const maxCount = Math.max(...sppList.map(s => s.count), 1);

      let html = `
        <div style="display: grid; grid-template-columns: 45px 1fr 140px 140px 180px 110px; gap: 12px; padding: 12px 16px; background: #f8fafc; font-size: 11px; font-weight: 700; color: var(--text-dim); text-transform: uppercase; border-bottom: 1px solid var(--border);">
          <div>Rank</div>
          <div>Species</div>
          <div style="text-align: right;">False Alarms</div>
          <div style="text-align: right;">Avg Confidence</div>
          <div>Primary Trigger</div>
          <div style="text-align: right;">Action</div>
        </div>
      `;

      sppList.forEach((sp, idx) => {{
        const barWidth = Math.max((sp.count / maxCount) * 100, 6);
        const scName = sp.scientific_name || formatScientific(sp.species);
        
        let rankBadgeBg = '#f1f5f9';
        let rankColor = 'var(--text-muted)';
        if (idx === 0) {{ rankBadgeBg = '#fef08a'; rankColor = '#854d0e'; }}
        else if (idx === 1) {{ rankBadgeBg = '#e2e8f0'; rankColor = '#475569'; }}
        else if (idx === 2) {{ rankBadgeBg = '#ffedd5'; rankColor = '#9a3412'; }}

        html += `
          <div class="species-row" onclick="filterBySpeciesName('${{sp.species}}')" title="Click to filter gallery to ${{scName}} (${{sp.count}} frames)">
            <div style="display: flex; align-items: center;">
              <span style="font-size: 11px; font-weight: 800; background: ${{rankBadgeBg}}; color: ${{rankColor}}; width: 26px; height: 26px; border-radius: 50%; display: flex; align-items: center; justify-content: center;">
                #${{idx + 1}}
              </span>
            </div>

            <div>
              <div style="font-weight: 800; font-size: 14px; color: var(--text); display: flex; align-items: center; gap: 6px;">
                <span>${{sp.emoji}}</span>
                <span style="font-style: italic;">${{scName}}</span>
              </div>
              <div style="background: #f1f5f9; border-radius: 4px; height: 6px; width: 100%; max-width: 260px; overflow: hidden; margin-top: 6px;">
                <div style="width: ${{barWidth}}%; background: #e11d48; height: 100%; border-radius: 4px;"></div>
              </div>
            </div>

            <div style="text-align: right;">
              <div style="font-weight: 800; font-size: 15px; color: #be123c;">${{sp.count}} frames</div>
              <div style="font-size: 11px; color: var(--text-muted);">${{sp.pct_str}} of errors</div>
            </div>

            <div style="text-align: right;">
              <div style="font-weight: 800; font-size: 14px; color: var(--text);">${{sp.mean_conf_str}}</div>
              <div style="font-size: 10px; color: var(--text-dim);">median: ${{sp.median_conf_str}}</div>
            </div>

            <div>
              <span style="font-size: 11px; font-weight: 600; background: #eff6ff; color: #1e40af; padding: 3px 8px; border-radius: 6px; border: 1px solid #bfdbfe; display: inline-block;">
                ${{sp.top_bg}}
              </span>
              <div style="font-size: 10px; color: var(--text-dim); margin-top: 2px;">${{sp.top_bg_pct}} of errors (Cam ${{sp.top_camera}})</div>
            </div>

            <div style="text-align: right; display: flex; align-items: center; justify-content: flex-end;">
              <span class="species-filter-btn">
                Filter Gallery &rarr;
              </span>
            </div>
          </div>
        `;
      }});

      container.innerHTML = html;
    }}

    function filterBySpeciesName(species) {{
      activeCameraFilter = '';
      document.querySelectorAll('.bar-col').forEach(b => b.classList.remove('active'));
      document.getElementById('cam-filter').value = '';
      document.getElementById('bg-filter').value = '';
      document.getElementById('species-filter').value = species;

      document.getElementById('conf-slider').value = 0;
      document.getElementById('conf-val').innerText = '0%';

      const sppData = dataStore.species || {{}};
      const list = sppData[currentModel] || sppData['fine_tuned'] || [];
      const found = list.find(s => s.species === species);
      const scName = found ? (found.scientific_name || formatScientific(found.species)) : formatScientific(species);
      const emoji = found ? found.emoji : '🐾';
      const displayName = `${{emoji}} ${{scName}}`;
      const count = found ? found.count : '';

      const banner = document.getElementById('active-filter-banner');
      banner.classList.add('active');
      document.getElementById('active-filter-text').innerText = `Filtering by Hallucinated Species: ${{displayName}} (${{count}} frames across all locations)`;

      filterGallery();
      document.getElementById('gallery-section').scrollIntoView({{ behavior: 'smooth' }});
    }}

    const DIAGNOSTIC_RATIONALES = {{
      'panthera onca': {{
        'Night / IR Flash': 'Monochrome infrared flash creates high-contrast circular reflections on waxy leaves and tree bark that strongly mimic the circular rosette coat patterns of Panthera onca.',
        'Dense Vegetation': 'Patchy daytime sunlight penetrating dense canopy creates mottled dappled shadows on the forest floor matching Panthera onca coat camouflage.'
      }},
      'leopardus pardalis': {{
        'Night / IR Flash': 'Smaller high-frequency flash reflections and leaf specular highlights activate convolutional filters tuned for Leopardus pardalis spots and rosettes.'
      }},
      'crax rubra': {{
        'Night / IR Flash': 'Dark foliage silhouettes against illuminated ground mimic the heavy, dark body profile of Crax rubra roosting or foraging.',
        'Dense Vegetation': 'Dense understory brush with rounded dark leaf clusters mimics the dark plumage contours of Crax rubra.'
      }},
      'equus ferus': {{
        'Dense Vegetation': 'Vertical tree trunks and hanging lianas in daytime forest scenes match the silhouettes of large ungulate legs and equine backs of Equus ferus.'
      }},
      'meleagris ocellata': {{
        'Dense Vegetation': 'Low leaf mounds, dry branch clusters, and ground debris create textured contours mimicking the speckled plumage of Meleagris ocellata.'
      }},
      'urocyon cinereoargenteus': {{
        'Dense Vegetation': 'Low-lying horizontal logs and leafy brush match the slender profile and bushy tail silhouette of Urocyon cinereoargenteus.'
      }},
      'dasyprocta punctata': {{
        'Night / IR Flash': 'Small rounded stones or leaf piles illuminated by flash mimic the compact hunched body of Dasyprocta punctata.',
        'Dense Vegetation': 'Leaf litter and root knots on the forest floor mimic the dark reddish-brown contours of Dasyprocta punctata.'
      }},
      'madoqua guentheri': {{
        'Night / IR Flash': 'Thin vertical twigs and sharp specular highlights mimic the delicate legs and pointed ears of Madoqua guentheri in monochrome night vision.'
      }},
      'aepyceros melampus': {{
        'Open / Diffuse Landscape': 'Open grassy fields and horizon contours trigger the network’s learned environmental prior for Aepyceros melampus.',
        'High Contrast / Lighting': 'Bright sunlight over dry savanna terrain creates sharp contrast contours matching Aepyceros melampus silhouettes.'
      }},
      'bos taurus': {{
        'High Contrast / Lighting': 'Overexposed bright sunlit patches and broad horizontal clearing contours mimic the broad, light-colored flanks of Bos taurus.'
      }}
    }};

    let currentMatrixMode = 'count';

    function setMatrixMode(mode) {{
      currentMatrixMode = mode;
      document.getElementById('btn-matrix-count').style.background = mode === 'count' ? 'var(--primary)' : '#f1f5f9';
      document.getElementById('btn-matrix-count').style.color = mode === 'count' ? '#ffffff' : 'var(--text)';
      document.getElementById('btn-matrix-pct').style.background = mode === 'pct' ? 'var(--primary)' : '#f1f5f9';
      document.getElementById('btn-matrix-pct').style.color = mode === 'pct' ? '#ffffff' : 'var(--text)';
      renderConfusionMatrix();
    }}

    // Render RQ3 Confusion Matrix Table
    function renderConfusionMatrix() {{
      const matrixObj = dataStore.matrix || {{}};
      const m = matrixObj[currentModel] || matrixObj['fine_tuned'] || matrixObj;
      if (!m || !m.species || !m.rows) return;
      const table = document.getElementById('confusion-matrix-table');

      let html = '<thead><tr><th class="row-hdr">Background / Species</th>';
      m.species.forEach(sp => {{
        html += `<th style="text-transform: capitalize;">${{sp}}</th>`;
      }});
      html += '<th>Row Total</th></tr></thead><tbody>';

      m.rows.forEach(r => {{
        html += `<tr><th class="row-hdr">${{r.bg}}</th>`;
        r.cells.forEach(c => {{
          const isZero = c.count === 0;
          const isSelected = activeMatrixCell && activeMatrixCell.bg === r.bg && activeMatrixCell.species === c.species;

          let displayVal = currentMatrixMode === 'pct' ? (isZero ? '—' : c.pct_str) : (isZero ? '0' : c.count);
          let bgStyle = '';
          if (isZero) {{
            bgStyle = 'background: #ffffff; color: #cbd5e1;';
          }} else if (c.pct_val >= 0.20 || c.count >= 14) {{
            bgStyle = 'background: #1d4ed8; color: #ffffff;';
          }} else if (c.pct_val >= 0.10 || c.count >= 8) {{
            bgStyle = 'background: #3b82f6; color: #ffffff;';
          }} else if (c.pct_val >= 0.04 || c.count >= 4) {{
            bgStyle = 'background: #93c5fd; color: #0f172a;';
          }} else {{
            bgStyle = 'background: #dbeafe; color: #1e40af;';
          }}

          html += `
            <td class="matrix-cell ${{isZero ? 'zero' : ''}} ${{isSelected ? 'active' : ''}}" 
                style="${{bgStyle}}" 
                onclick="inspectMatrixCell('${{r.bg}}', '${{c.species}}')" 
                title="${{r.bg}} × ${{c.species}}: ${{c.count}} frames (${{c.pct_str}} of ${{r.bg}} false alarms)">
              ${{displayVal}}
            </td>
          `;
        }});
        html += `<td style="font-weight: 800; background: #f8fafc;">${{r.total}}</td></tr>`;
      }});
      html += '</tbody>';
      table.innerHTML = html;

      // Render the inspector card below table
      renderMatrixInspector();
    }}

    function inspectMatrixCell(bg, species) {{
      activeMatrixCell = {{ bg, species }};
      renderConfusionMatrix();
    }}

    function renderMatrixInspector() {{
      const container = document.getElementById('matrix-inspector');
      if (!container) return;
      if (!activeMatrixCell) {{
        container.innerHTML = '';
        return;
      }}

      const bg = activeMatrixCell.bg;
      const species = activeMatrixCell.species;
      const allItems = dataStore[currentModel] || [];

      const topSppList = [
        'panthera onca', 'leopardus pardalis', 'crax rubra', 'equus ferus',
        'meleagris ocellata', 'urocyon cinereoargenteus', 'dasyprocta punctata',
        'madoqua guentheri', 'aepyceros melampus', 'bos taurus'
      ];

      let matching = [];
      if (species === 'other species') {{
        matching = allItems.filter(it => it.bg === bg && !topSppList.includes(it.species));
      }} else {{
        matching = allItems.filter(it => it.bg === bg && it.species === species);
      }}

      if (matching.length === 0) {{
        container.innerHTML = `
          <div class="inspector-card">
            <div class="inspector-header">
              <div class="inspector-title">
                <span>🔍 Hotspot Inspector:</span>
                <span style="color: var(--primary); text-transform: capitalize;">${{species}}</span>
                <span style="font-size: 12px; font-weight: 600; color: var(--text-muted);">&bull; ${{bg}}</span>
              </div>
              <span class="pane-badge" style="background: #f1f5f9; color: var(--text-dim); border: none;">0 Frames Recorded</span>
            </div>
            <div class="inspector-desc" style="border-left-color: #94a3b8;">
              <strong>No Hallucinations Observed:</strong> The model never misclassified empty scenes under <em>${{bg}}</em> as <em>${{species}}</em>. Environmental textures in this background do not trigger the feature detectors for this species.
            </div>
          </div>
        `;
        return;
      }}

      const camCounts = {{}};
      matching.forEach(it => {{
        camCounts[it.camera] = (camCounts[it.camera] || 0) + 1;
      }});
      const camPills = Object.entries(camCounts)
        .sort((a, b) => b[1] - a[1])
        .map(([c, cnt]) => `<span style="font-size: 11px; background: #e2e8f0; padding: 2px 8px; border-radius: 4px; font-weight: 700;">Cam ${{c}}: ${{cnt}} (${{Math.round((cnt / matching.length) * 100)}}%)</span>`)
        .join(' ');

      let explanation = DIAGNOSTIC_RATIONALES[species] && DIAGNOSTIC_RATIONALES[species][bg];
      if (!explanation) {{
        explanation = `Visual background features characteristic of <b>${{bg}}</b> activate feature maps associated with <b>${{species}}</b> in the absence of any true animal.`;
      }}

      const previewThumbs = matching.slice(0, 8).map(it => `
        <div class="inspector-thumb-card" onclick="openModalById('${{it.id}}')" title="Click to inspect frame details | Camera ${{it.camera}}">
          <img src="${{it.thumb}}" alt="${{it.species}}" loading="lazy">
          <div class="inspector-thumb-meta">
            <span style="font-weight: 700;">Cam ${{it.camera}}</span>
            <span style="color: var(--primary); font-weight: 800;">${{it.confidence_pct}}</span>
          </div>
        </div>
      `).join('');

      container.innerHTML = `
        <div class="inspector-card">
          <div class="inspector-header">
            <div class="inspector-title">
              <span style="font-size: 18px;">🎯</span>
              <span>Hotspot Inspector:</span>
              <span style="color: var(--primary); text-transform: capitalize;">${{species}}</span>
              <span style="font-size: 12px; font-weight: 600; color: var(--text-muted);">&bull; ${{bg}}</span>
            </div>
            <div style="display: flex; gap: 8px; align-items: center;">
              <span class="pane-badge" style="font-size: 12px; padding: 4px 12px;">${{matching.length}} False Alarm Frames</span>
              <button class="nav-btn" onclick="filterByMatrixCell('${{bg}}', '${{species}}')" style="background: var(--primary); color: #ffffff; font-weight: 700; padding: 5px 12px; font-size: 11px;">
                🔍 View all ${{matching.length}} in Gallery Below &darr;
              </button>
            </div>
          </div>
          
          <div class="inspector-desc">
            <strong>Diagnostic Root Cause (RQ3):</strong> ${{explanation}}
          </div>

          <div style="margin-bottom: 12px; display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
            <span style="font-size: 11px; font-weight: 700; color: var(--text-dim); text-transform: uppercase;">Camera Locations:</span>
            ${{camPills}}
          </div>

          <div style="font-size: 11px; font-weight: 700; color: var(--text-muted); margin-bottom: 8px; text-transform: uppercase;">
            Example Hallucinated Frames (Click Any Image to Inspect Top-5 Probabilities):
          </div>
          <div class="inspector-grid">
            ${{previewThumbs}}
          </div>
        </div>
      `;
    }}

    // Filter by clicking a cell in the confusion matrix
    function filterByMatrixCell(bg, species) {{
      // 1. Clear conflicting camera filter
      activeCameraFilter = '';
      document.querySelectorAll('.bar-col').forEach(b => b.classList.remove('active'));
      document.getElementById('cam-filter').value = '';

      // 2. Reset confidence threshold to show all matching frames
      document.getElementById('conf-slider').value = 0;
      document.getElementById('conf-val').innerText = '0%';

      // 3. Set background and species
      document.getElementById('bg-filter').value = bg;
      if (species === 'other species') {{
        document.getElementById('species-filter').value = '';
      }} else {{
        document.getElementById('species-filter').value = species;
      }}

      // 4. Update banner
      const banner = document.getElementById('active-filter-banner');
      banner.classList.add('active');
      document.getElementById('active-filter-text').innerText = `Matrix Filter: "${{bg}}" × "${{species}}" (All cameras)`;

      filterGallery();
      document.getElementById('gallery-container').scrollIntoView({{ behavior: 'smooth' }});
    }}

    function openModalById(id) {{
      const idx = currentFiltered.findIndex(it => it.id === id);
      if (idx !== -1) {{
        openModalByIndex(idx);
      }} else {{
        const item = (dataStore[currentModel] || []).find(it => it.id === id);
        if (item) {{
          const scientificName = item.scientific_name || formatScientific(item.species);
          document.getElementById('modal-title').innerHTML = `${{item.emoji || '🐾'}} <span style="font-style: italic;">${{scientificName}}</span>`;
          document.getElementById('modal-sub').innerText = `Camera ${{item.camera}} | Seq: ${{item.seq_id}} | Background: ${{item.bg}} | Confidence: ${{item.confidence_pct}}`;
          document.getElementById('modal-img').src = item.thumb;
          const top5Rows = (item.top5 || []).map(t => {{
            const sc = formatScientific(t.species);
            return `
            <tr>
              <td style="width: 25px; color: var(--text-dim); font-weight: 700;">#${{t.rank}}</td>
              <td style="font-weight: ${{t.is_empty ? '800' : '500'}}; color: ${{t.is_empty ? 'var(--accent-green)' : 'var(--text)'}};">
                <span style="font-style: italic;">${{sc}}</span> ${{t.is_empty ? '(GROUND TRUTH: EMPTY)' : ''}}
              </td>
              <td style="width: 80px; text-align: right; font-family: monospace; font-weight: 700;">${{t.prob_pct}}</td>
              <td style="width: 140px;">
                <div style="background: #e2e8f0; border-radius: 4px; height: 8px; overflow: hidden;">
                  <div style="width: ${{t.prob * 100}}%; background: ${{t.is_empty ? 'var(--accent-green)' : 'var(--primary)'}}; height: 100%;"></div>
                </div>
              </td>
            </tr>
          `;
          }}).join('');
          document.getElementById('modal-top5').innerHTML = `<tbody>${{top5Rows}}</tbody>`;
          document.getElementById('modal').classList.add('active');
        }}
      }}
    }}

    // Camera selection (from bar chart)
    function selectCamera(camId) {{
      activeCameraFilter = camId;
      document.querySelectorAll('.bar-col').forEach(b => b.classList.remove('active'));

      if (camId) {{
        const b = document.getElementById('barchart-cam-' + camId);
        if (b) b.classList.add('active');
        
        const banner = document.getElementById('active-filter-banner');
        banner.classList.add('active');
        document.getElementById('active-filter-text').innerText = `Filtering by Camera ${{camId}} (${{getCamFpCount(camId)}} false detections)`;
      }} else {{
        document.getElementById('active-filter-banner').classList.remove('active');
      }}

      document.getElementById('cam-filter').value = camId;
      filterGallery();
      document.getElementById('gallery-section').scrollIntoView({{ behavior: 'smooth' }});
    }}

    function getCamFpCount(camId) {{
      const found = (dataStore.cameras || []).find(c => c.camera === camId);
      return found ? found.n_fp : 0;
    }}

    function resetAllFilters() {{
      activeCameraFilter = '';
      document.getElementById('cam-filter').value = '';
      document.getElementById('species-filter').value = '';
      document.getElementById('bg-filter').value = '';
      document.getElementById('conf-slider').value = 0;
      document.getElementById('conf-val').innerText = '0%';
      document.getElementById('active-filter-banner').classList.remove('active');
      document.querySelectorAll('.bar-col').forEach(b => b.classList.remove('active'));

      filterGallery();
    }}

    function switchModel(modelKey) {{
      currentModel = modelKey;
      document.querySelectorAll('.model-btn').forEach(b => b.classList.remove('active'));
      event.target.classList.add('active');
      renderSpeciesStats();
      renderConfusionMatrix();
      filterGallery();
    }}

    const SPECIES_COLOR_MAP = {{
      'panthera onca': {{ bg: '#fef3c7', bgLight: '#fffbeb', border: '#fde68a', accent: '#d97706', text: '#92400e' }},
      'leopardus pardalis': {{ bg: '#ffedd5', bgLight: '#fff7ed', border: '#fed7aa', accent: '#ea580c', text: '#9a3412' }},
      'equus ferus': {{ bg: '#f3e8ff', bgLight: '#faf5ff', border: '#e9d5ff', accent: '#7c3aed', text: '#5b21b6' }},
      'crax rubra': {{ bg: '#e2e8f0', bgLight: '#f1f5f9', border: '#cbd5e1', accent: '#475569', text: '#1e293b' }},
      'dasyprocta punctata': {{ bg: '#ccfbf1', bgLight: '#f0fdfa', border: '#99f6e4', accent: '#0d9488', text: '#115e59' }},
      'meleagris ocellata': {{ bg: '#ffe4e6', bgLight: '#fff1f2', border: '#fecdd3', accent: '#e11d48', text: '#9f1239' }},
      'madoqua guentheri': {{ bg: '#e0f2fe', bgLight: '#f0f9ff', border: '#bae6fd', accent: '#0284c7', text: '#0369a1' }},
      'urocyon cinereoargenteus': {{ bg: '#fef9c3', bgLight: '#fefce8', border: '#fef08a', accent: '#ca8a04', text: '#854d0e' }},
      'motorcycle': {{ bg: '#e4e4e7', bgLight: '#f4f4f5', border: '#d4d4d8', accent: '#71717a', text: '#3f3f46' }},
      'argusianus argus': {{ bg: '#e0e7ff', bgLight: '#eef2ff', border: '#c7d2fe', accent: '#4f46e5', text: '#3730a3' }},
      'aepyceros melampus': {{ bg: '#dcfce7', bgLight: '#f0fdf4', border: '#bbf7d0', accent: '#16a34a', text: '#166534' }},
      'giraffa camelopardalis': {{ bg: '#fef3c7', bgLight: '#fffbeb', border: '#fde68a', accent: '#d97706', text: '#78350f' }},
      'tapirus bairdii': {{ bg: '#ede9fe', bgLight: '#f5f3ff', border: '#ddd6fe', accent: '#8b5cf6', text: '#6d28d9' }},
      'bos taurus': {{ bg: '#fee2e2', bgLight: '#fef2f2', border: '#fca5a5', accent: '#dc2626', text: '#991b1b' }},
      'loxodonta africana': {{ bg: '#e2e8f0', bgLight: '#f8fafc', border: '#cbd5e1', accent: '#64748b', text: '#334155' }},
      'paguma larvata': {{ bg: '#d1fae5', bgLight: '#ecfdf5', border: '#a7f3d0', accent: '#059669', text: '#065f46' }},
      'crocuta crocuta': {{ bg: '#ffedd5', bgLight: '#fff7ed', border: '#fed7aa', accent: '#ea580c', text: '#7c2d12' }},
      'nasua narica': {{ bg: '#fef3c7', bgLight: '#fffbeb', border: '#fde68a', accent: '#b45309', text: '#78350f' }},
    }};

    function getSpeciesColor(sp) {{
      if (SPECIES_COLOR_MAP[sp]) return SPECIES_COLOR_MAP[sp];
      const fallbacks = [
        {{ bg: '#fef3c7', bgLight: '#fffbeb', border: '#fde68a', accent: '#d97706', text: '#92400e' }},
        {{ bg: '#f3e8ff', bgLight: '#faf5ff', border: '#e9d5ff', accent: '#7c3aed', text: '#5b21b6' }},
        {{ bg: '#ccfbf1', bgLight: '#f0fdfa', border: '#99f6e4', accent: '#0d9488', text: '#115e59' }},
        {{ bg: '#ffe4e6', bgLight: '#fff1f2', border: '#fecdd3', accent: '#e11d48', text: '#9f1239' }},
        {{ bg: '#e0f2fe', bgLight: '#f0f9ff', border: '#bae6fd', accent: '#0284c7', text: '#0369a1' }},
        {{ bg: '#dcfce7', bgLight: '#f0fdf4', border: '#bbf7d0', accent: '#16a34a', text: '#166534' }}
      ];
      let hash = 0;
      for (let i = 0; i < (sp || '').length; i++) hash = sp.charCodeAt(i) + ((hash << 5) - hash);
      return fallbacks[Math.abs(hash) % fallbacks.length];
    }}

    const CAMERA_COLOR_MAP = {{
      '529': {{ bg: '#fff1f2', border: '#fecdd3', text: '#9f1239', sub: '#be123c', tagBg: '#ffe4e6', tagText: '#881337' }},
      '547': {{ bg: '#eff6ff', border: '#bfdbfe', text: '#1e3a8a', sub: '#1d4ed8', tagBg: '#dbeafe', tagText: '#1e40af' }},
      '456': {{ bg: '#f5f3ff', border: '#ddd6fe', text: '#4c1d95', sub: '#6d28d9', tagBg: '#ede9fe', tagText: '#5b21b6' }},
      '249': {{ bg: '#ecfdf5', border: '#a7f3d0', text: '#064e3b', sub: '#047857', tagBg: '#d1fae5', tagText: '#065f46' }},
      '109': {{ bg: '#fffbeb', border: '#fde68a', text: '#78350f', sub: '#b45309', tagBg: '#fef3c7', tagText: '#92400e' }},
      '8':   {{ bg: '#ecfeff', border: '#a5f3fc', text: '#164e63', sub: '#0e7490', tagBg: '#cffafe', tagText: '#155e75' }},
      '275': {{ bg: '#fdf2f8', border: '#fbcfe8', text: '#831843', sub: '#be185d', tagBg: '#fce7f3', tagText: '#9d174d' }},
      '375': {{ bg: '#fefce8', border: '#fef08a', text: '#713f12', sub: '#ca8a04', tagBg: '#fef9c3', tagText: '#854d0e' }},
      '474': {{ bg: '#f0fdf4', border: '#bbf7d0', text: '#14532d', sub: '#15803d', tagBg: '#dcfce7', tagText: '#166534' }},
      '469': {{ bg: '#f8fafc', border: '#cbd5e1', text: '#0f172a', sub: '#334155', tagBg: '#e2e8f0', tagText: '#1e293b' }},
      '524': {{ bg: '#fff7ed', border: '#fed7aa', text: '#7c2d12', sub: '#ea580c', tagBg: '#ffedd5', tagText: '#9a3412' }},
    }};

    function getCameraColor(camId) {{
      const c = String(camId || '');
      if (CAMERA_COLOR_MAP[c]) return CAMERA_COLOR_MAP[c];
      const fallbacks = [
        {{ bg: '#eff6ff', border: '#bfdbfe', text: '#1e3a8a', sub: '#1d4ed8', tagBg: '#dbeafe', tagText: '#1e40af' }},
        {{ bg: '#f5f3ff', border: '#ddd6fe', text: '#4c1d95', sub: '#6d28d9', tagBg: '#ede9fe', tagText: '#5b21b6' }},
        {{ bg: '#ecfdf5', border: '#a7f3d0', text: '#064e3b', sub: '#047857', tagBg: '#d1fae5', tagText: '#065f46' }},
        {{ bg: '#fffbeb', border: '#fde68a', text: '#78350f', sub: '#b45309', tagBg: '#fef3c7', tagText: '#92400e' }},
        {{ bg: '#fff1f2', border: '#fecdd3', text: '#9f1239', sub: '#be123c', tagBg: '#ffe4e6', tagText: '#881337' }},
      ];
      let hash = 0;
      for (let i = 0; i < c.length; i++) hash = c.charCodeAt(i) + ((hash << 5) - hash);
      return fallbacks[Math.abs(hash) % fallbacks.length];
    }}

    function renderCard(item, idx) {{
      let badgeClass = 'badge-low';
      if (item.confidence >= 0.80) badgeClass = 'badge-high';
      else if (item.confidence >= 0.50) badgeClass = 'badge-med';

      const spCol = getSpeciesColor(item.species);
      const camCol = getCameraColor(item.camera);
      const emoji = item.emoji || '🐾';
      const scientificName = item.scientific_name || formatScientific(item.species);

      return `
        <div class="card" onclick="openModalByIndex(${{idx}})" style="border-color: ${{spCol.border}};">
          <!-- Upper area: Background behind image colored by SPECIES -->
          <div class="card-img-wrap" style="background: ${{spCol.bg}};" title="Species: ${{scientificName}}">
            <div class="card-img-inner">
              <img src="${{item.thumb}}" alt="${{item.species}}" loading="lazy">
              <span class="badge ${{badgeClass}}" title="The AI model was ${{item.confidence_pct}} certain this empty photo showed ${{scientificName}}">
                AI Conf: ${{item.confidence_pct}}
              </span>
              <span class="badge-bg">${{item.bg}}</span>
            </div>
          </div>

          <div class="card-body">
            <!-- Species label & clear AI certainty statement -->
            <div class="card-species-header" style="background: ${{spCol.bgLight}}; border-left: 3px solid ${{spCol.accent}};">
              <div class="card-species" style="color: ${{spCol.text}};">
                <span>${{emoji}}</span>
                <span style="font-style: italic;">${{scientificName}}</span>
              </div>
              <div class="card-conf-expl">
                AI hallucinated on empty photo (<b>${{item.confidence_pct}}</b> certain)
              </div>
            </div>

            <!-- Below part: Background colored by the CAMERA that took the photo -->
            <div class="card-camera-box" style="background: ${{camCol.bg}}; border: 1px solid ${{camCol.border}};">
              <div class="card-meta">
                <span style="color: ${{camCol.text}}; font-weight: 800; display: flex; align-items: center; gap: 4px;">
                  <span>📷</span>
                  <span>Camera ${{item.camera}}</span>
                </span>
                <span style="color: ${{camCol.sub}}; font-size: 10px; font-family: monospace;">Seq: ${{item.seq_id.substring(0, 8)}}...</span>
              </div>
              <div class="card-tags">
                <span class="tag" style="background: ${{camCol.tagBg}}; color: ${{camCol.tagText}};">Lum: ${{item.luminance}}</span>
                <span class="tag" style="background: ${{camCol.tagBg}}; color: ${{camCol.tagText}};">Green: ${{item.greenness}}</span>
              </div>
            </div>
          </div>
        </div>
      `;
    }}

    function filterGallery() {{
      const items = dataStore[currentModel] || [];
      const sp = document.getElementById('species-filter').value;
      const bg = document.getElementById('bg-filter').value;
      const cam = document.getElementById('cam-filter').value;
      const minConf = parseFloat(document.getElementById('conf-slider').value) / 100.0;
      const sortBy = document.getElementById('sort-select').value;

      currentFiltered = items.filter(item => {{
        if (sp && item.species !== sp) return false;
        if (bg && item.bg !== bg) return false;
        if (cam && item.camera !== cam) return false;
        if (item.confidence < minConf) return false;
        return true;
      }});

      currentFiltered.sort((a, b) => {{
        if (sortBy === 'conf_desc') return b.confidence - a.confidence;
        if (sortBy === 'conf_asc') return a.confidence - b.confidence;
        if (sortBy === 'cam') return parseInt(a.camera) - parseInt(b.camera);
        if (sortBy === 'species') return a.species.localeCompare(b.species);
        if (sortBy === 'lum_desc') return b.luminance - a.luminance;
        if (sortBy === 'green_desc') return b.greenness - a.greenness;
        return 0;
      }});

      currentPage = 1;
      renderGalleryPage();
    }}

    function renderGalleryPage() {{
      const container = document.getElementById('gallery-container');
      const paginationContainer = document.getElementById('gallery-pagination');
      const totalItems = currentFiltered.length;

      document.getElementById('stat-count').innerText = totalItems;

      if (totalItems === 0) {{
        container.innerHTML = `<div class="empty-state"><h3>No frames match the active filters</h3><p style="margin-top: 6px;">Try clearing species or camera filters to view more images.</p></div>`;
        document.getElementById('stat-displayed').innerText = '0';
        if (paginationContainer) paginationContainer.innerHTML = '';
        return;
      }}

      const pageSize = currentPageSize === 'all' ? totalItems : currentPageSize;
      const totalPages = Math.ceil(totalItems / pageSize) || 1;

      if (currentPage > totalPages) currentPage = totalPages;
      if (currentPage < 1) currentPage = 1;

      const startIndex = (currentPage - 1) * pageSize;
      const endIndex = Math.min(startIndex + pageSize, totalItems);

      if (currentPageSize === 'all' || totalItems <= pageSize) {{
        document.getElementById('stat-displayed').innerText = `${{totalItems}}`;
      }} else {{
        document.getElementById('stat-displayed').innerText = `${{startIndex + 1}}–${{endIndex}}`;
      }}

      const visibleItems = currentFiltered.slice(startIndex, endIndex);
      container.innerHTML = visibleItems.map((it, localIdx) => {{
        const globalIdx = startIndex + localIdx;
        return renderCard(it, globalIdx);
      }}).join('');

      if (!paginationContainer) return;
      if (currentPageSize === 'all' || totalPages <= 1) {{
        paginationContainer.innerHTML = '';
        return;
      }}

      let pagHtml = `
        <button class="pagination-btn" onclick="prevPage()" ${{currentPage === 1 ? 'disabled' : ''}}>
          &larr; Previous
        </button>
      `;

      for (let p = 1; p <= totalPages; p++) {{
        if (totalPages > 7) {{
          if (p !== 1 && p !== totalPages && Math.abs(p - currentPage) > 2) {{
            if (p === 2 || p === totalPages - 1) {{
              pagHtml += `<span style="color: var(--text-dim); padding: 0 4px; font-weight: 700;">...</span>`;
            }}
            continue;
          }}
        }}
        const isActive = p === currentPage ? ' active' : '';
        pagHtml += `
          <button class="pagination-btn${{isActive}}" onclick="goToPage(${{p}})">
            ${{p}}
          </button>
        `;
      }}

      pagHtml += `
        <button class="pagination-btn" onclick="nextPage()" ${{currentPage === totalPages ? 'disabled' : ''}}>
          Next &rarr;
        </button>
        <span class="pagination-info">Page ${{currentPage}} of ${{totalPages}} (${{totalItems}} frames)</span>
      `;

      paginationContainer.innerHTML = pagHtml;
    }}

    function updateConf(val) {{
      document.getElementById('conf-val').innerText = val + '%';
      filterGallery();
    }}

    function openModalByIndex(idx) {{
      if (idx < 0 || idx >= currentFiltered.length) return;
      currentModalIndex = idx;
      const item = currentFiltered[idx];

      const scientificName = item.scientific_name || formatScientific(item.species);
      document.getElementById('modal-title').innerHTML = `${{item.emoji || '🐾'}} <span style="font-style: italic;">${{scientificName}}</span>`;
      document.getElementById('modal-sub').innerText = `Camera ${{item.camera}} | Seq: ${{item.seq_id}} | Background: ${{item.bg}} | Confidence: ${{item.confidence_pct}} (Frame ${{idx + 1}} of ${{currentFiltered.length}})`;
      document.getElementById('modal-img').src = item.thumb;

      const top5Rows = (item.top5 || []).map(t => {{
        const sc = formatScientific(t.species);
        return `
        <tr>
          <td style="width: 25px; color: var(--text-dim); font-weight: 700;">#${{t.rank}}</td>
          <td style="font-weight: ${{t.is_empty ? '800' : '500'}}; color: ${{t.is_empty ? 'var(--accent-green)' : 'var(--text)'}};">
            <span style="font-style: italic;">${{sc}}</span> ${{t.is_empty ? '(GROUND TRUTH: EMPTY)' : ''}}
          </td>
          <td style="width: 80px; text-align: right; font-family: monospace; font-weight: 700;">${{t.prob_pct}}</td>
          <td style="width: 140px;">
            <div style="background: #e2e8f0; height: 8px; border-radius: 4px; overflow: hidden;">
              <div class="prob-bar" style="width: ${{t.prob * 100}}%; background: ${{t.is_empty ? 'var(--accent-green)' : 'var(--primary)'}};"></div>
            </div>
          </td>
        </tr>
      `;
      }}).join('');

      document.getElementById('modal-top5').innerHTML = `
        <thead>
          <tr>
            <th>Rank</th>
            <th>Species Class</th>
            <th style="text-align: right;">Class Probability</th>
            <th>Probability Bar</th>
          </tr>
        </thead>
        <tbody>${{top5Rows}}</tbody>
      `;

      document.getElementById('modal').classList.add('active');
    }}

    function prevModalItem() {{
      if (currentModalIndex > 0) openModalByIndex(currentModalIndex - 1);
    }}

    function nextModalItem() {{
      if (currentModalIndex < currentFiltered.length - 1) openModalByIndex(currentModalIndex + 1);
    }}

    function closeModal() {{
      document.getElementById('modal').classList.remove('active');
    }}

    // Keyboard navigation
    document.addEventListener('keydown', (e) => {{
      const modal = document.getElementById('modal');
      if (modal.classList.contains('active')) {{
        if (e.key === 'ArrowLeft') prevModalItem();
        if (e.key === 'ArrowRight') nextModalItem();
        if (e.key === 'Escape') closeModal();
      }}
    }});

    // Initialize all visual components
    renderCameraBarChart();
    renderSpeciesStats();
    renderConfusionMatrix();
    filterGallery();
  </script>
</body>
</html>
"""
    output_path.write_text(html_content, encoding='utf-8')
    print(f"Enhanced False Detection Gallery written to: {output_path.resolve()}")
    return str(output_path)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Idea 4: Audit empty camera-trap hallucinations & build False Detection Gallery."
    )
    parser.add_argument(
        "--split",
        default="val",
        help="WILDS split to evaluate: 'val' (OOD, default), 'id_val', or 'all'",
    )
    parser.add_argument(
        "--output-html",
        default=None,
        help="Path to save the interactive gallery HTML (defaults to phase1/pair_a/figures/false_detection_gallery.html)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  Idea 4: Camera-Trap Empty Image Hallucination Audit")
    print(f"  Split: {args.split} (unseen out-of-distribution camera locations)")
    print("=" * 70)

    print("\n[1/3] Loading predictions for Fine-Tuned ResNet-50 & Linear Probe...")
    data = load_comparative_data(split=args.split, compute_bg_features=True)
    ft_fp = data["ft_fp"]
    probe_fp = data["probe_fp"]
    true_empty = data["true_empty"]

    print(f"  - Total ground-truth empty frames: {len(true_empty):,}")
    print(f"  - Fine-Tuned False Detections: {len(ft_fp):,} ({len(ft_fp) / len(true_empty):.2%} FPR)")
    print(f"  - Linear Probe False Detections: {len(probe_fp):,} ({len(probe_fp) / len(true_empty):.2%} FPR)")
    print(f"  - Model Disagreements on empty frames: {len(data['disagreements']):,} frames")

    print("\n[2/3] Analyzing Camera Hotspots & Location Contextual Shortcuts...")
    cam_stats = true_empty.groupby("location").size().rename("n_empty")
    cam_fp = ft_fp.groupby("location").agg(
        n_fp=("id", "size"),
        mean_conf=("confidence", "mean"),
        top_species=("prediction_name", lambda s: s.mode()[0]),
    )
    cam_merged = pd.concat([cam_stats, cam_fp], axis=1).fillna({"n_fp": 0, "mean_conf": 0.0, "top_species": "None"})
    cam_merged["fpr"] = cam_merged["n_fp"] / cam_merged["n_empty"]
    cam_top = cam_merged.sort_values("n_fp", ascending=False).head(5)
    print("  Top 5 Cameras with highest false detection counts:")
    for loc, r in cam_top.iterrows():
        print(f"    Camera {loc:3d}: {int(r['n_fp'])} / {int(r['n_empty'])} FPs ({r['fpr']:.1%}) | Top: {r['top_species']} (Conf: {r['mean_conf']:.1%})")

    burst_res = analyze_burst_persistence(ft_fp, true_empty)
    print(f"\n[3/3] Evaluating Burst Sequence Dynamics...")
    print(f"  - In {burst_res['full_hallucination_rate']:.1%} of affected bursts, ALL empty frames are hallucinated!")

    print("\nBuilding Interactive Multi-Model False Detection Gallery with Camera Hotspots...")
    out_path = Path(args.output_html) if args.output_html else resolve("phase1/pair_a/figures/false_detection_gallery.html")
    build_false_detection_gallery_html(data, output_path=out_path)

    print("\n" + "=" * 70)
    print("  AUDIT COMPLETE!")
    print(f"  Interactive Gallery: {out_path.resolve()}")
    print("  To view the gallery:")
    print("    - In VS Code: Right-click false_detection_gallery.html -> 'Show Preview'")
    print("    - In Browser: Open the file directly in Chrome / Firefox")
    print("=" * 70)


if __name__ == "__main__":
    main()
