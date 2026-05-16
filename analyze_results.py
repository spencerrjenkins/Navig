#!/usr/bin/env python3
"""Comprehensive analysis of stage-6 model comparison results.

Sections produced:
  1. Overall comparison table (GeoScore, distance, accuracy at 5 thresholds)
  2. Distance distribution (percentiles)
  3. Evidence usage breakdown — does using each evidence type actually help?
  4. Geographic breakdown — performance by continent
  5. Parse failure analysis
  6. Model agreement — when do models agree vs. disagree?
  7. Top hard / easy images across all models

Usage — auto-discover mode (recommended after merge_all_shards.py):
    python analyze_results.py \\
        --dir output/im2gps3k_rgb_images \\
        --output output/im2gps3k_rgb_images/comparison_report.txt

    Scans every *_merged/ subdirectory for results_s6_*.jsonl files and
    includes all of them.  Labels are derived from the filename stem
    (e.g. results_s6_deepseek.jsonl → "deepseek").

Usage — explicit mode:
    python analyze_results.py \\
        --files output/.../cmp_shard_deepseek_merged/results_s6_deepseek.jsonl \\
                output/.../cmp_shard_falcon_merged/results_s6_falcon.jsonl \\
        --labels DeepSeek-7B Falcon-11B \\
        --output output/im2gps3k_rgb_images/comparison_report.txt
"""

import argparse
import io
import json
import re
import sys
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from collections import defaultdict
from pathlib import Path

# ── Canonical model registry ──────────────────────────────────────────────────
# Defines the 7 canonical models for --canonical mode.
# Models missing on disk are skipped with a warning.
BASE_DIR = "output/im2gps3k_rgb_images"
CANONICAL_MODELS = [
    {"path": f"{BASE_DIR}/llava_merged/results_s6_llava.jsonl",                              "label": "LLaVA-1.6"},
    {"path": f"{BASE_DIR}/cmp_shard_deepseek_merged/results_s6_deepseek.jsonl",              "label": "DeepSeek-7B"},
    {"path": f"{BASE_DIR}/cmp_shard_llama32vision_merged/results_s6_llama32vision.jsonl",    "label": "LLaMA-3.2-11B"},
    {"path": f"{BASE_DIR}/cmp_shard_cpm_merged/results_s6_cpm.jsonl",                        "label": "MiniCPM-V-2.6"},
    {"path": f"{BASE_DIR}/guess_merged/results_s6_llama32.jsonl",                            "label": "LLaMA-3.2-11B (swap)"},
    {"path": f"{BASE_DIR}/cmp_shard_falcon_merged/results_s6_falcon.jsonl",                  "label": "Falcon-11B"},
    {"path": f"{BASE_DIR}/cmp_shard_qwen_merged/results_s6_qwen.jsonl",                      "label": "Qwen2-VL-7B"},
]

# ── Geo utilities ─────────────────────────────────────────────────────────────

def haversine(c1, c2):
    lat1, lon1 = map(radians, c1)
    lat2, lon2 = map(radians, c2)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 6371 * 2 * atan2(sqrt(a), sqrt(1 - a))


def geoscore(d):
    return 5000 * np.exp(-d / 1492.7)


def _parse_coord(value):
    """Parse coordinate from plain float, degree-notation ('8.64° N'), or
    hedged string ('Approximately 75° W'). Raises ValueError for 'Unknown' etc."""
    s = str(value).strip()
    if s.lower() in ('unknown', 'n/a', ''):
        raise ValueError(f"Unparseable coordinate: {s!r}")
    s = re.sub(r'(?i)approximately\s*', '', s).strip()
    m = re.match(r'^(-?\d+(?:\.\d+)?)\s*°?\s*([NSEWnsew])?$', s)
    if m:
        val = float(m.group(1))
        if (m.group(2) or '').upper() in ('S', 'W'):
            val = -val
        return val
    return float(s)


def get_dist(row):
    correct = [float(row['LAT']), float(row['LON'])]
    try:
        pred = [_parse_coord(row['answer']['latitude']), _parse_coord(row['answer']['longitude'])]
        return haversine(pred, correct), False
    except Exception:
        return 10000.0, True


def lat_to_continent(lat, lon):
    """Rough continent assignment from lat/lon."""
    if lat > 66.5:
        return 'Arctic'
    if lat < -60:
        return 'Antarctica'
    if -35 < lat < 37 and -20 < lon < 55:
        return 'Africa'
    if lat > 35 and -30 < lon < 60:
        return 'Europe'
    if lat > 0 and 60 < lon < 180:
        return 'Asia (N)'
    if -10 < lat < 35 and 60 < lon < 180:
        return 'Asia (S/SE)'
    if -55 < lat < -10 and 110 < lon < 180:
        return 'Oceania'
    if 15 < lat < 75 and -170 < lon < -50:
        return 'N. America'
    if -60 < lat < 15 and -90 < lon < -30:
        return 'S. America'
    return 'Other'


# ── Data loading ──────────────────────────────────────────────────────────────

def load_results(path):
    rows = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    rows[row['ID']] = row
    except FileNotFoundError:
        return None
    return rows


# ── Scoring helpers ───────────────────────────────────────────────────────────

THRESHOLDS = [1, 25, 200, 750, 2500]
THR_NAMES  = ['@1km', '@25km', '@200km', '@750km', '@2500km']


def score_rows(rows_list):
    """Returns (distances, failures) for a list of rows."""
    dists, fails = [], 0
    for row in rows_list:
        d, f = get_dist(row)
        dists.append(d)
        fails += f
    return dists, fails


def summary_stats(dists, n_total):
    n = len(dists)
    accs = [sum(d <= t for d in dists) / n_total for t in THRESHOLDS]
    return {
        'n': n_total,
        'geoscore': float(np.mean([geoscore(d) for d in dists])),
        'avg_dist': float(np.mean(dists)),
        'median_dist': float(np.median(dists)),
        'p25': float(np.percentile(dists, 25)),
        'p75': float(np.percentile(dists, 75)),
        'p90': float(np.percentile(dists, 90)),
        'p95': float(np.percentile(dists, 95)),
        'accs': accs,
    }


# ── Report sections ───────────────────────────────────────────────────────────

def section(out, title):
    out.write('\n' + '=' * 80 + '\n')
    out.write(f'  {title}\n')
    out.write('=' * 80 + '\n')


def sec1_overall(out, models):
    section(out, '1. OVERALL COMPARISON')
    col = max(len(m['label']) for m in models) + 2
    header = (f"{'Model':<{col}} {'N':>5}  {'GeoScore':>9}  {'ΔGS':>7}  "
              f"{'AvgDist':>9}  {'MedDist':>9}  "
              + '  '.join(f'{n:>7}' for n in THR_NAMES)
              + f"  {'Fail%':>6}")
    out.write(header + '\n')
    out.write('-' * len(header) + '\n')

    baseline_gs = models[0]['stats']['geoscore']
    for m in models:
        s = m['stats']
        delta = s['geoscore'] - baseline_gs
        delta_str = f'{delta:+.1f}' if delta != 0.0 else '  base'
        acc_str = '  '.join(f'{a:>7.4f}' for a in s['accs'])
        fail_pct = 100.0 * m['failures'] / s['n']
        out.write(
            f"{m['label']:<{col}} {s['n']:>5}  {s['geoscore']:>9.2f}  {delta_str:>7}  "
            f"{s['avg_dist']:>9.1f}  {s['median_dist']:>9.1f}  "
            f"{acc_str}  {fail_pct:>5.1f}%\n"
        )


def sec2_distribution(out, models):
    section(out, '2. DISTANCE DISTRIBUTION (km)')
    col = max(len(m['label']) for m in models) + 2
    header = f"{'Model':<{col}} {'P25':>8}  {'P50':>8}  {'P75':>8}  {'P90':>8}  {'P95':>8}"
    out.write(header + '\n')
    out.write('-' * len(header) + '\n')
    for m in models:
        s = m['stats']
        out.write(
            f"{m['label']:<{col}} {s['p25']:>8.1f}  {s['median_dist']:>8.1f}  "
            f"{s['p75']:>8.1f}  {s['p90']:>8.1f}  {s['p95']:>8.1f}\n"
        )


def sec3_evidence(out, models):
    section(out, '3. EVIDENCE USAGE — DOES EACH TYPE HELP?')
    out.write('  GeoScore when evidence type is present (1) vs. absent (0)\n\n')

    ev_types = ['reasoning', 'osm', 'rag', 'comment']
    col = max(len(m['label']) for m in models) + 2

    for ev in ev_types:
        out.write(f'  {ev.upper()}\n')
        hdr = f"  {'Model':<{col}} {'N(1)':>6}  {'GS(1)':>8}  {'N(0)':>6}  {'GS(0)':>8}  {'Delta':>8}\n"
        out.write(hdr)
        out.write('  ' + '-' * (len(hdr) - 3) + '\n')
        for m in models:
            present, absent = [], []
            for row in m['rows'].values():
                usage = row.get('usage', {})
                d, _ = get_dist(row)
                if usage.get(ev, 0):
                    present.append(d)
                else:
                    absent.append(d)
            gs1 = np.mean([geoscore(d) for d in present]) if present else float('nan')
            gs0 = np.mean([geoscore(d) for d in absent])  if absent  else float('nan')
            delta = gs1 - gs0
            out.write(
                f"  {m['label']:<{col}} {len(present):>6}  {gs1:>8.2f}  "
                f"{len(absent):>6}  {gs0:>8.2f}  {delta:>+8.2f}\n"
            )
        out.write('\n')


def sec4_geographic(out, models, common_ids):
    section(out, '4. GEOGRAPHIC BREAKDOWN (GeoScore by continent)')

    # Get continents for common IDs using the first model's rows
    first_rows = models[0]['rows']
    continents = defaultdict(list)
    for img_id in common_ids:
        row = first_rows[img_id]
        cont = lat_to_continent(float(row['LAT']), float(row['LON']))
        continents[cont].append(img_id)

    col_label = max(len(m['label']) for m in models) + 2
    cont_list = sorted(continents.keys(), key=lambda c: -len(continents[c]))

    # Header
    col_cont = max(len(c) for c in cont_list) + 2
    hdr_parts = [f"{'Continent':<{col_cont}}", f"{'N':>5}"]
    for m in models:
        hdr_parts.append(f"{m['label'][:10]:>11}")
    out.write('  ' + '  '.join(hdr_parts) + '\n')
    out.write('  ' + '-' * (sum(len(p) + 2 for p in hdr_parts)) + '\n')

    for cont in cont_list:
        ids = continents[cont]
        row_parts = [f"{cont:<{col_cont}}", f"{len(ids):>5}"]
        for m in models:
            dists = [get_dist(m['rows'][i])[0] for i in ids if i in m['rows']]
            gs = np.mean([geoscore(d) for d in dists]) if dists else float('nan')
            row_parts.append(f"{gs:>11.1f}")
        out.write('  ' + '  '.join(row_parts) + '\n')


def sec5_failures(out, models):
    section(out, '5. PARSE FAILURE ANALYSIS')
    out.write('  (Parse failures penalised as 10000 km in all other sections)\n\n')
    col = max(len(m['label']) for m in models) + 2
    hdr = f"  {'Model':<{col}} {'N':>5}  {'Failures':>9}  {'Fail%':>7}  {'GS excl. failures':>18}\n"
    out.write(hdr)
    out.write('  ' + '-' * (len(hdr) - 3) + '\n')
    for m in models:
        success_dists = []
        for row in m['rows'].values():
            d, fail = get_dist(row)
            if not fail:
                success_dists.append(d)
        gs_excl = np.mean([geoscore(d) for d in success_dists]) if success_dists else float('nan')
        out.write(
            f"  {m['label']:<{col}} {m['stats']['n']:>5}  {m['failures']:>9}  "
            f"{100*m['failures']/m['stats']['n']:>6.1f}%  {gs_excl:>18.2f}\n"
        )


def sec6_agreement(out, models, common_ids):
    section(out, '6. MODEL AGREEMENT')
    out.write('  How often do models predict within X km of each other?\n\n')
    thresholds = [50, 200, 500]

    pairs = [(i, j) for i in range(len(models)) for j in range(i+1, len(models))]
    col = 25

    hdr = f"  {'Pair':<{col}}"
    for t in thresholds:
        hdr += f"  {'within '+str(t)+'km':>12}"
    out.write(hdr + '\n')
    out.write('  ' + '-' * (len(hdr) - 2) + '\n')

    for i, j in pairs:
        ma, mb = models[i], models[j]
        label = f"{ma['label'][:10]} vs {mb['label'][:10]}"
        counts = [0] * len(thresholds)
        n = 0
        for img_id in common_ids:
            ra = ma['rows'].get(img_id)
            rb = mb['rows'].get(img_id)
            if ra is None or rb is None:
                continue
            try:
                pred_a = [_parse_coord(ra['answer']['latitude']), _parse_coord(ra['answer']['longitude'])]
                pred_b = [_parse_coord(rb['answer']['latitude']), _parse_coord(rb['answer']['longitude'])]
                d = haversine(pred_a, pred_b)
                for k, t in enumerate(thresholds):
                    if d <= t:
                        counts[k] += 1
                n += 1
            except Exception:
                pass
        row_str = f"  {label:<{col}}"
        for c in counts:
            row_str += f"  {c/n if n else 0:>11.3f} "
        out.write(row_str + '\n')


def sec7_hard_easy(out, models, common_ids, k=15):
    section(out, f'7. HARDEST AND EASIEST IMAGES (avg distance across all models, top {k})')

    avg_dists = {}
    for img_id in common_ids:
        dists = []
        for m in models:
            if img_id in m['rows']:
                d, _ = get_dist(m['rows'][img_id])
                dists.append(d)
        if dists:
            avg_dists[img_id] = np.mean(dists)

    sorted_ids = sorted(avg_dists, key=lambda x: avg_dists[x])
    col_id = min(max(len(i) for i in common_ids[:5]), 40)

    out.write(f'\n  EASIEST (all models predict well)\n')
    hdr = f"  {'Image ID':<{col_id}}  {'AvgDist':>9}  " + '  '.join(f"{m['label'][:9]:>9}" for m in models) + '\n'
    out.write(hdr)
    out.write('  ' + '-' * (len(hdr) - 3) + '\n')
    for img_id in sorted_ids[:k]:
        row_parts = [f"  {img_id[:col_id]:<{col_id}}  {avg_dists[img_id]:>9.1f}"]
        for m in models:
            if img_id in m['rows']:
                d, _ = get_dist(m['rows'][img_id])
                row_parts.append(f"{d:>9.1f}")
            else:
                row_parts.append(f"{'N/A':>9}")
        out.write('  '.join(row_parts) + '\n')

    out.write(f'\n  HARDEST (all models struggle)\n')
    out.write(hdr)
    out.write('  ' + '-' * (len(hdr) - 3) + '\n')
    for img_id in sorted_ids[-k:][::-1]:
        row_parts = [f"  {img_id[:col_id]:<{col_id}}  {avg_dists[img_id]:>9.1f}"]
        for m in models:
            if img_id in m['rows']:
                d, _ = get_dist(m['rows'][img_id])
                row_parts.append(f"{d:>9.1f}")
            else:
                row_parts.append(f"{'N/A':>9}")
        out.write('  '.join(row_parts) + '\n')


# ── Auto-discovery ────────────────────────────────────────────────────────────

def discover_files(directory):
    """Return (files, labels) for every results_s6_*.jsonl in *_merged/ dirs.

    Labels are the file stem with 'results_s6_' stripped.  When the same
    label would appear more than once (e.g. both llava_merged/ and
    llava_shard_merged/ contain results_s6_llava.jsonl), the merged
    directory name is prepended to make each label unique.
    """
    root = Path(directory)
    if not root.is_dir():
        print(f'ERROR: {root} is not a directory', file=sys.stderr)
        sys.exit(1)

    # Collect (path, base_label, dir_name) triples
    triples = []
    for merged_dir in sorted(d for d in root.iterdir()
                             if d.is_dir() and d.name.endswith('_merged')):
        for f in sorted(merged_dir.glob('results_s6_*.jsonl')):
            base_label = f.stem[len('results_s6_'):]
            triples.append((str(f), base_label, merged_dir.name))

    if not triples:
        print(f'No results_s6_*.jsonl found in *_merged/ subdirs of {root}',
              file=sys.stderr)
        sys.exit(1)

    # Disambiguate duplicate base labels
    from collections import Counter
    counts = Counter(t[1] for t in triples)
    files, labels = [], []
    for path, base_label, dir_name in triples:
        label = f'{dir_name}/{base_label}' if counts[base_label] > 1 else base_label
        files.append(path)
        labels.append(label)
    return files, labels


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--canonical', action='store_true',
                     help='Use the built-in 7-model canonical registry')
    src.add_argument('--dir', type=str,
                     help='Directory to scan for *_merged/results_s6_*.jsonl')
    src.add_argument('--files', nargs='+',
                     help='Explicit merged result JSONL files, one per model')
    p.add_argument('--labels', nargs='*',
                   help='Display names (must match --files count; ignored with --dir/--canonical)')
    p.add_argument('--output', type=str, default=None,
                   help='Save report to this file (also always prints to stdout)')
    args = p.parse_args()

    if args.canonical:
        files  = [m['path']  for m in CANONICAL_MODELS]
        labels = [m['label'] for m in CANONICAL_MODELS]
        print(f'Using canonical {len(CANONICAL_MODELS)}-model registry:')
        for f, l in zip(files, labels):
            exists = '✓' if Path(f).exists() else '✗ MISSING'
            print(f'  [{l}]  {f}  {exists}')
    elif args.dir:
        files, labels = discover_files(args.dir)
        if args.labels:
            print('WARNING: --labels is ignored when --dir is used.', file=sys.stderr)
        print(f'Auto-discovered {len(files)} result file(s):')
        for f, l in zip(files, labels):
            print(f'  [{l}]  {f}')
    else:
        files = args.files
        if args.labels and len(args.labels) != len(files):
            print('ERROR: --labels count must match --files count.', file=sys.stderr)
            sys.exit(1)
        labels = args.labels or [Path(f).stem.replace('results_s6_', '') for f in files]

    models = []
    for path, label in zip(files, labels):
        rows = load_results(path)
        if rows is None:
            print(f'WARNING: {path} not found, skipping.', file=sys.stderr)
            continue
        dists, failures = score_rows(list(rows.values()))
        models.append({
            'label': label,
            'rows': rows,
            'dists': dists,
            'failures': failures,
            'stats': summary_stats(dists, len(rows)),
        })

    if not models:
        print('No valid result files found.')
        sys.exit(1)

    # Sort by GeoScore descending
    models.sort(key=lambda m: m['stats']['geoscore'], reverse=True)

    # Common IDs present in all models
    common_ids = sorted(
        set.intersection(*[set(m['rows'].keys()) for m in models])
    )
    print(f'Common images across all models: {len(common_ids)}')

    buf = io.StringIO()

    buf.write(f'NAVIG — Stage-6 Comparison Report ({len(models)} models)\n')
    buf.write(f'Models: {", ".join(m["label"] for m in models)}\n')
    buf.write(f'Common samples: {len(common_ids)}\n')

    sec1_overall(buf, models)
    sec2_distribution(buf, models)
    sec3_evidence(buf, models)
    sec4_geographic(buf, models, common_ids)
    sec5_failures(buf, models)
    sec6_agreement(buf, models, common_ids)
    sec7_hard_easy(buf, models, common_ids)

    report = buf.getvalue()
    print(report)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f'\nReport saved to: {args.output}')


if __name__ == '__main__':
    main()
