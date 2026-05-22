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

Usage — auto-discover mode (recommended after merge_all_shards.py)::

    python analysis/analyze_results.py \\
        --dir output/im2gps3k \\
        --output output/im2gps3k/comparison_report.txt

Usage — explicit mode::

    python analysis/analyze_results.py \\
        --files output/.../results_s6_deepseek.jsonl \\
                output/.../results_s6_falcon.jsonl \\
        --labels DeepSeek-7B Falcon-11B
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import io
import json
import re
from collections import Counter, defaultdict

import numpy as np

from metrics import (
    geoscore,
    haversine_distance,
    parse_coord,
    THRESHOLDS,
    THRESHOLD_NAMES,
)

# ── Canonical model registry ───────────────────────────────────────────────────
BASE_DIR = "output/im2gps3k"
CANONICAL_MODELS = [
    # Zero-shot models (no NAVIG SFT adapter)
    {
        "path": f"{BASE_DIR}/cmp_shard_llama32vision_merged/results_s6_llama32vision.jsonl",
        "label": "LLaMA-3.2-11B",
    },
    {
        "path": f"{BASE_DIR}/cmp_shard_deepseek_merged/results_s6_deepseek.jsonl",
        "label": "DeepSeek-7B",
    },
    {
        "path": f"{BASE_DIR}/cmp_shard_falcon_merged/results_s6_falcon.jsonl",
        "label": "Falcon-11B",
    },
    # Base models — no NAVIG LoRA for stage-1 reasoning
    {
        "path": f"{BASE_DIR}/cmp_shard_llava_merged/results_s6_llava.jsonl",
        "label": "LLaVA-1.6",
    },
    {
        "path": f"{BASE_DIR}/cmp_shard_cpm_merged/results_s6_cpm.jsonl",
        "label": "MiniCPM-V-2.6",
    },
    {
        "path": f"{BASE_DIR}/cmp_shard_qwen_merged/results_s6_qwen.jsonl",
        "label": "Qwen2.5-VL-7B",
    },
    # SFT models — NAVIG LoRA adapter for stage-1 reasoning
    {
        "path": f"{BASE_DIR}/cmp_shard_llava_sft_merged/results_s6_llava_sft.jsonl",
        "label": "LLaVA-1.6 (SFT)",
    },
    {
        "path": f"{BASE_DIR}/cmp_shard_cpm_sft_merged/results_s6_cpm_sft.jsonl",
        "label": "MiniCPM-V-2.6 (SFT)",
    },
    {
        "path": f"{BASE_DIR}/cmp_shard_qwen_sft_merged/results_s6_qwen_sft.jsonl",
        "label": "Qwen2.5-VL-7B (SFT)",
    },
    # Swap experiment: LLaMA-3.2-11B runs only stage 6 on existing pipeline evidence
    {
        "path": f"{BASE_DIR}/guess_merged/results_s6_llama32vision.jsonl",
        "label": "LLaMA-3.2-11B (swap)",
    },
]

THR_NAMES = [f"@{t}km" for t in THRESHOLDS]


# ── Geo utilities ──────────────────────────────────────────────────────────────


def get_dist(row: dict) -> tuple[float, bool]:
    correct = [float(row["LAT"]), float(row["LON"])]
    try:
        pred = [
            parse_coord(row["answer"]["latitude"]),
            parse_coord(row["answer"]["longitude"]),
        ]
        return haversine_distance(pred, correct), False
    except Exception:
        return 10_000.0, True


from geo_utils import lat_to_continent  # noqa: E402


# ── Data loading ───────────────────────────────────────────────────────────────


def load_results(path: str) -> dict[str, dict] | None:
    try:
        rows = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    rows[row["ID"]] = row
        return rows
    except FileNotFoundError:
        return None


# ── Scoring helpers ────────────────────────────────────────────────────────────


def score_rows(rows_list: list[dict]) -> tuple[list[float], int]:
    dists, fails = [], 0
    for row in rows_list:
        d, f = get_dist(row)
        dists.append(d)
        fails += f
    return dists, fails


def summary_stats(dists: list[float], n_total: int) -> dict:
    accs = [sum(d <= t for d in dists) / n_total for t in THRESHOLDS]
    return {
        "n": n_total,
        "geoscore": float(np.mean([geoscore(d) for d in dists])),
        "avg_dist": float(np.mean(dists)),
        "median_dist": float(np.median(dists)),
        "p25": float(np.percentile(dists, 25)),
        "p75": float(np.percentile(dists, 75)),
        "p90": float(np.percentile(dists, 90)),
        "p95": float(np.percentile(dists, 95)),
        "accs": accs,
    }


# ── Report sections ────────────────────────────────────────────────────────────


def section(out, title: str) -> None:
    out.write("\n" + "=" * 80 + "\n")
    out.write(f"  {title}\n")
    out.write("=" * 80 + "\n")


def sec1_overall(out, models: list[dict]) -> None:
    section(out, "1. OVERALL COMPARISON")
    col = max(len(m["label"]) for m in models) + 2
    header = (
        f"{'Model':<{col}} {'N':>5}  {'GeoScore':>9}  {'ΔGS':>7}  "
        f"{'AvgDist':>9}  {'MedDist':>9}  "
        + "  ".join(f"{n:>7}" for n in THR_NAMES)
        + f"  {'Fail%':>6}"
    )
    out.write(header + "\n")
    out.write("-" * len(header) + "\n")
    baseline_gs = models[0]["stats"]["geoscore"]
    for m in models:
        s = m["stats"]
        delta = s["geoscore"] - baseline_gs
        delta_str = f"{delta:+.1f}" if delta != 0.0 else "  base"
        acc_str = "  ".join(f"{a:>7.4f}" for a in s["accs"])
        fail_pct = 100.0 * m["failures"] / s["n"]
        out.write(
            f"{m['label']:<{col}} {s['n']:>5}  {s['geoscore']:>9.2f}  {delta_str:>7}  "
            f"{s['avg_dist']:>9.1f}  {s['median_dist']:>9.1f}  "
            f"{acc_str}  {fail_pct:>5.1f}%\n"
        )


def sec2_distribution(out, models: list[dict]) -> None:
    section(out, "2. DISTANCE DISTRIBUTION (km)")
    col = max(len(m["label"]) for m in models) + 2
    header = (
        f"{'Model':<{col}} {'P25':>8}  {'P50':>8}  {'P75':>8}  {'P90':>8}  {'P95':>8}"
    )
    out.write(header + "\n")
    out.write("-" * len(header) + "\n")
    for m in models:
        s = m["stats"]
        out.write(
            f"{m['label']:<{col}} {s['p25']:>8.1f}  {s['median_dist']:>8.1f}  "
            f"{s['p75']:>8.1f}  {s['p90']:>8.1f}  {s['p95']:>8.1f}\n"
        )


def sec3_evidence(out, models: list[dict]) -> None:
    section(out, "3. EVIDENCE USAGE — DOES EACH TYPE HELP?")
    out.write("  GeoScore when evidence type is present (1) vs. absent (0)\n\n")
    ev_types = ["reasoning", "osm", "rag", "comment"]
    col = max(len(m["label"]) for m in models) + 2
    for ev in ev_types:
        out.write(f"  {ev.upper()}\n")
        hdr = f"  {'Model':<{col}} {'N(1)':>6}  {'GS(1)':>8}  {'N(0)':>6}  {'GS(0)':>8}  {'Delta':>8}\n"
        out.write(hdr)
        out.write("  " + "-" * (len(hdr) - 3) + "\n")
        for m in models:
            present, absent = [], []
            for row in m["rows"].values():
                d, _ = get_dist(row)
                if row.get("usage", {}).get(ev, 0):
                    present.append(d)
                else:
                    absent.append(d)
            gs1 = np.mean([geoscore(d) for d in present]) if present else float("nan")
            gs0 = np.mean([geoscore(d) for d in absent]) if absent else float("nan")
            delta = gs1 - gs0
            out.write(
                f"  {m['label']:<{col}} {len(present):>6}  {gs1:>8.2f}  "
                f"{len(absent):>6}  {gs0:>8.2f}  {delta:>+8.2f}\n"
            )
        out.write("\n")


def sec4_geographic(out, models: list[dict], common_ids: list[str], use_api: bool = True) -> None:
    section(out, "4. GEOGRAPHIC BREAKDOWN (GeoScore by continent)")
    first_rows = models[0]["rows"]
    continents: dict[str, list[str]] = defaultdict(list)
    for img_id in common_ids:
        row = first_rows[img_id]
        cont = lat_to_continent(float(row["LAT"]), float(row["LON"]), use_api=use_api)
        continents[cont].append(img_id)

    col_label = max(len(m["label"]) for m in models) + 2
    cont_list = sorted(continents.keys(), key=lambda c: -len(continents[c]))
    col_cont = max(len(c) for c in cont_list) + 2

    hdr_parts = [f"{'Continent':<{col_cont}}", f"{'N':>5}"]
    for m in models:
        hdr_parts.append(f"{m['label'][:10]:>11}")
    out.write("  " + "  ".join(hdr_parts) + "\n")
    out.write("  " + "-" * (sum(len(p) + 2 for p in hdr_parts)) + "\n")

    for cont in cont_list:
        ids = continents[cont]
        row_parts = [f"{cont:<{col_cont}}", f"{len(ids):>5}"]
        for m in models:
            dists = [get_dist(m["rows"][i])[0] for i in ids if i in m["rows"]]
            gs = np.mean([geoscore(d) for d in dists]) if dists else float("nan")
            row_parts.append(f"{gs:>11.1f}")
        out.write("  " + "  ".join(row_parts) + "\n")


def sec5_failures(out, models: list[dict]) -> None:
    section(out, "5. PARSE FAILURE ANALYSIS")
    out.write("  (Parse failures penalised as 10 000 km in all other sections)\n\n")
    col = max(len(m["label"]) for m in models) + 2
    hdr = f"  {'Model':<{col}} {'N':>5}  {'Failures':>9}  {'Fail%':>7}  {'GS excl. failures':>18}\n"
    out.write(hdr)
    out.write("  " + "-" * (len(hdr) - 3) + "\n")
    for m in models:
        success_dists = [
            get_dist(row)[0] for row in m["rows"].values() if not get_dist(row)[1]
        ]
        gs_excl = (
            np.mean([geoscore(d) for d in success_dists])
            if success_dists
            else float("nan")
        )
        out.write(
            f"  {m['label']:<{col}} {m['stats']['n']:>5}  {m['failures']:>9}  "
            f"{100*m['failures']/m['stats']['n']:>6.1f}%  {gs_excl:>18.2f}\n"
        )


def sec6_agreement(out, models: list[dict], common_ids: list[str]) -> None:
    section(out, "6. MODEL AGREEMENT")
    out.write("  How often do models predict within X km of each other?\n\n")
    thresholds = [50, 200, 500]
    pairs = [(i, j) for i in range(len(models)) for j in range(i + 1, len(models))]
    col = 25
    hdr = f"  {'Pair':<{col}}"
    for t in thresholds:
        hdr += f"  {'within ' + str(t) + 'km':>12}"
    out.write(hdr + "\n")
    out.write("  " + "-" * (len(hdr) - 2) + "\n")
    for i, j in pairs:
        ma, mb = models[i], models[j]
        label = f"{ma['label'][:10]} vs {mb['label'][:10]}"
        counts = [0] * len(thresholds)
        n = 0
        for img_id in common_ids:
            ra, rb = ma["rows"].get(img_id), mb["rows"].get(img_id)
            if ra is None or rb is None:
                continue
            try:
                pred_a = [
                    parse_coord(ra["answer"]["latitude"]),
                    parse_coord(ra["answer"]["longitude"]),
                ]
                pred_b = [
                    parse_coord(rb["answer"]["latitude"]),
                    parse_coord(rb["answer"]["longitude"]),
                ]
                d = haversine_distance(pred_a, pred_b)
                for k, t in enumerate(thresholds):
                    if d <= t:
                        counts[k] += 1
                n += 1
            except Exception:
                pass
        row_str = f"  {label:<{col}}"
        for c in counts:
            row_str += f"  {c/n if n else 0:>11.3f} "
        out.write(row_str + "\n")


def sec7_hard_easy(out, models: list[dict], common_ids: list[str], k: int = 15) -> None:
    section(
        out, f"7. HARDEST AND EASIEST IMAGES (avg distance across all models, top {k})"
    )
    avg_dists = {
        img_id: np.mean(
            [get_dist(m["rows"][img_id])[0] for m in models if img_id in m["rows"]]
        )
        for img_id in common_ids
    }
    sorted_ids = sorted(avg_dists, key=lambda x: avg_dists[x])
    col_id = min(max(len(i) for i in common_ids[:5]), 40)

    for title, ids in [("EASIEST", sorted_ids[:k]), ("HARDEST", sorted_ids[-k:][::-1])]:
        out.write(
            f"\n  {title} (all models predict {'well' if title=='EASIEST' else 'poorly'})\n"
        )
        hdr = (
            f"  {'Image ID':<{col_id}}  {'AvgDist':>9}  "
            + "  ".join(f"{m['label'][:9]:>9}" for m in models)
            + "\n"
        )
        out.write(hdr)
        out.write("  " + "-" * (len(hdr) - 3) + "\n")
        for img_id in ids:
            parts = [f"  {img_id[:col_id]:<{col_id}}  {avg_dists[img_id]:>9.1f}"]
            for m in models:
                if img_id in m["rows"]:
                    d, _ = get_dist(m["rows"][img_id])
                    parts.append(f"{d:>9.1f}")
                else:
                    parts.append(f"{'N/A':>9}")
            out.write("  ".join(parts) + "\n")


def sec8_pairwise(out, models: list[dict], common_ids: list[str], top_k: int = 10) -> None:
    section(out, "8. PAIRWISE HEAD-TO-HEAD")

    if len(models) < 2:
        out.write("  Need at least 2 models for pairwise comparison.\n")
        return

    col_id = min(max(len(i) for i in common_ids), 40) if common_ids else 20

    # Win/loss/tie summary for all pairs
    col_pair = 33
    hdr = (
        f"  {'Pair (A  →  B)':<{col_pair}} {'N':>5}  "
        f"{'A wins%':>8}  {'B wins%':>8}  {'Tied%':>8}  "
        f"{'GS_A':>8}  {'GS_B':>8}  {'ΔGS':>7}\n"
    )
    out.write(hdr)
    out.write("  " + "-" * (len(hdr) - 3) + "\n")

    pairs = [
        (i, j)
        for i in range(len(models))
        for j in range(i + 1, len(models))
    ]
    for i, j in pairs:
        ma, mb = models[i], models[j]
        deltas = []
        for img_id in common_ids:
            if img_id not in ma["rows"] or img_id not in mb["rows"]:
                continue
            da, _ = get_dist(ma["rows"][img_id])
            db, _ = get_dist(mb["rows"][img_id])
            deltas.append(db - da)  # positive → A closer than B

        n = len(deltas)
        if n == 0:
            continue

        a_wins = sum(1 for d in deltas if d > 0.5)
        b_wins = sum(1 for d in deltas if d < -0.5)
        tied = n - a_wins - b_wins
        gs_a = ma["stats"]["geoscore"]
        gs_b = mb["stats"]["geoscore"]
        pair_label = f"{ma['label'][:14]} → {mb['label'][:14]}"
        out.write(
            f"  {pair_label:<{col_pair}} {n:>5}  "
            f"{100*a_wins/n:>7.1f}%  {100*b_wins/n:>7.1f}%  {100*tied/n:>7.1f}%  "
            f"{gs_a:>8.2f}  {gs_b:>8.2f}  {gs_b - gs_a:>+7.2f}\n"
        )

    # Per-model quartiles and top-K vs. the top-ranked baseline
    baseline = models[0]
    for m in models[1:]:
        out.write(f"\n  ── {m['label']} vs. {baseline['label']} (baseline) ──\n")

        deltas: list[float] = []
        per_image: list[tuple[str, float, float, float]] = []

        for img_id in common_ids:
            if img_id not in baseline["rows"] or img_id not in m["rows"]:
                continue
            da, _ = get_dist(baseline["rows"][img_id])
            db, _ = get_dist(m["rows"][img_id])
            delta = db - da  # negative → m improved over baseline
            deltas.append(delta)
            per_image.append((img_id, da, db, delta))

        if not deltas:
            out.write("  No common images.\n")
            continue

        out.write(
            f"\n  Distance delta quartiles (km, negative = {m['label']} improved):\n"
        )
        for q, lbl in [(0, "min"), (25, "Q1"), (50, "median"), (75, "Q3"), (100, "max")]:
            out.write(f"    {lbl:6s}: {np.percentile(deltas, q):+.1f} km\n")

        per_image.sort(key=lambda x: x[3])

        out.write(f"\n  Top {top_k} most improved by {m['label']}:\n")
        out.write(
            f"  {'ID':<{col_id}} {'Baseline(km)':>13} {'Experiment(km)':>15} {'Delta':>10}\n"
        )
        for img_id, da, db, delta in per_image[:top_k]:
            out.write(f"  {img_id:<{col_id}} {da:>13.1f} {db:>15.1f} {delta:>+10.1f}\n")

        out.write(f"\n  Top {top_k} most hurt by {m['label']}:\n")
        out.write(
            f"  {'ID':<{col_id}} {'Baseline(km)':>13} {'Experiment(km)':>15} {'Delta':>10}\n"
        )
        for img_id, da, db, delta in per_image[-top_k:][::-1]:
            out.write(f"  {img_id:<{col_id}} {da:>13.1f} {db:>15.1f} {delta:>+10.1f}\n")


# ── Auto-discovery ─────────────────────────────────────────────────────────────


def discover_files(directory: str) -> tuple[list[str], list[str]]:
    root = Path(directory)
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory", file=sys.stderr)
        sys.exit(1)
    triples = []
    for merged_dir in sorted(
        d for d in root.iterdir() if d.is_dir() and d.name.endswith("_merged")
    ):
        for f in sorted(merged_dir.glob("results_s6_*.jsonl")):
            base_label = f.stem[len("results_s6_") :]
            triples.append((str(f), base_label, merged_dir.name))
    if not triples:
        print(
            f"No results_s6_*.jsonl found in *_merged/ subdirs of {root}",
            file=sys.stderr,
        )
        sys.exit(1)
    counts = Counter(t[1] for t in triples)
    files, labels = [], []
    for path, base_label, dir_name in triples:
        label = f"{dir_name}/{base_label}" if counts[base_label] > 1 else base_label
        files.append(path)
        labels.append(label)
    return files, labels


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--canonical",
        action="store_true",
        help="Use the built-in 7-model canonical registry",
    )
    src.add_argument(
        "--dir", type=str, help="Directory to scan for *_merged/results_s6_*.jsonl"
    )
    src.add_argument(
        "--files", nargs="+", help="Explicit merged result JSONL files, one per model"
    )
    p.add_argument(
        "--labels", nargs="*", help="Display names (must match --files count)"
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save report to this file (also always prints to stdout)",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Images to show in most-improved / most-hurt tables (default: 10)",
    )
    p.add_argument(
        "--no_api",
        action="store_true",
        help="Skip Nominatim reverse geocoding for continent assignment (use bounding-box fallback).",
    )
    args = p.parse_args()

    if args.canonical:
        files = [m["path"] for m in CANONICAL_MODELS]
        labels = [m["label"] for m in CANONICAL_MODELS]
        print(f"Using canonical {len(CANONICAL_MODELS)}-model registry:")
        for f, l in zip(files, labels):
            exists = "✓" if Path(f).exists() else "✗ MISSING"
            print(f"  [{l}]  {f}  {exists}")
    elif args.dir:
        files, labels = discover_files(args.dir)
        print(f"Auto-discovered {len(files)} result file(s):")
        for f, l in zip(files, labels):
            print(f"  [{l}]  {f}")
    else:
        files = args.files
        if args.labels and len(args.labels) != len(files):
            print("ERROR: --labels count must match --files count.", file=sys.stderr)
            sys.exit(1)
        labels = args.labels or [Path(f).stem.replace("results_s6_", "") for f in files]

    models = []
    for path, label in zip(files, labels):
        rows = load_results(path)
        if rows is None:
            print(f"WARNING: {path} not found, skipping.", file=sys.stderr)
            continue
        dists, failures = score_rows(list(rows.values()))
        models.append(
            {
                "label": label,
                "rows": rows,
                "dists": dists,
                "failures": failures,
                "stats": summary_stats(dists, len(rows)),
            }
        )

    if not models:
        print("No valid result files found.")
        sys.exit(1)

    models.sort(key=lambda m: m["stats"]["geoscore"], reverse=True)
    common_ids = sorted(set.intersection(*[set(m["rows"].keys()) for m in models]))
    print(f"Common images across all models: {len(common_ids)}")

    buf = io.StringIO()
    buf.write(f"NAVIG — Stage-6 Comparison Report ({len(models)} models)\n")
    buf.write(f"Models: {', '.join(m['label'] for m in models)}\n")
    buf.write(f"Common samples: {len(common_ids)}\n")

    sec1_overall(buf, models)
    sec2_distribution(buf, models)
    sec3_evidence(buf, models)
    sec4_geographic(buf, models, common_ids, use_api=not args.no_api)
    sec5_failures(buf, models)
    sec6_agreement(buf, models, common_ids)
    sec7_hard_easy(buf, models, common_ids)
    sec8_pairwise(buf, models, common_ids, top_k=args.top_k)

    report = buf.getvalue()
    print(report)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
