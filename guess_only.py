#!/usr/bin/env python3
"""Run only stage 6 (coordinate guessing) on an existing results_s5.jsonl.

This script is designed for the stage-6 swap experiment: keep all upstream
evidence from the original NAVIG pipeline and test whether a stronger guesser
model improves final accuracy, without re-running the expensive stages 1-5.

Typical workflow
----------------
1.  Run the full pipeline (stages 1-5) with the original llava model:
        sbatch script.sh          # produces results_s5.jsonl in each shard dir

2.  Merge the sharded s5 files:
        python merge_shards.py --base_dir output/im2gps3k_rgb_images \
            --num_shards 4 --results_file results_s5.jsonl \
            --output merged_s5.jsonl

3.  Run stage 6 with a new guesser (submit as a SLURM job via script_guess_only.sh):
        python guess_only.py \
            --s5_path output/im2gps3k_rgb_images/merged_s5.jsonl \
            --dataset_path dataset/im2gps3k_rgb_images \
            --model llama32vision \
            --model_path /fs/nexus-scratch/$USER/llama-3.2-11b-vision-instruct \
            --output output/im2gps3k_rgb_images/results_s6_llama32.jsonl

4.  Compare results:
        python compare_results.py \
            output/im2gps3k_rgb_images/results_s6_llava.jsonl \
            output/im2gps3k_rgb_images/results_s6_llama32.jsonl

Supported --model choices
--------------------------
  llava          LLaVA-1.6-Vicuna-7B (baseline, matches existing pipeline)
  qwen           Qwen2-VL-7B-Instruct (also works with Qwen2.5-VL weights)
  cpm            MiniCPM-V-2.6 (base, no fine-tuning)
  cpm_sft        MiniCPM-V-2.6 with NAVIG LoRA adapter (requires --ckpt_dir)
  llama32vision  Llama-3.2-11B-Vision-Instruct  [recommended for experiment]
  internvl2      InternVL2-8B
  deepseek       DeepSeek-VL-7B-Chat
  falcon         Falcon-11B-VLM
"""

import argparse
import json
import os
import sys
import re
from tqdm import tqdm
import numpy as np
import prompts
from utils import load_data, dump_jsonl, parse_json, haversine_distance


def Geoscore(distance):
    return 5000 * np.exp(-distance / 1492.7)


def load_model(model_name: str, model_path: str, ckpt_dir: str = None):
    from llm import LLaVA, Qwen, CPM, CPM_sft, Llama32Vision, InternVL2, DeepSeekVL, FalconVLM
    if model_name == 'cpm_sft':
        if not ckpt_dir:
            raise ValueError("--ckpt_dir is required for cpm_sft")
        dispatch = {'cpm_sft': lambda: CPM_sft(model_path=model_path, ckpt_dir=ckpt_dir)}
    else:
        dispatch = {
            'llava':         lambda: LLaVA(model_path=model_path),
            'qwen':          lambda: Qwen(model_path=model_path),
            'cpm':           lambda: CPM(model_path=model_path),
            'llama32vision': lambda: Llama32Vision(model_path=model_path),
            'internvl2':     lambda: InternVL2(model_path=model_path),
            'deepseek':      lambda: DeepSeekVL(model_path=model_path),
            'falcon':        lambda: FalconVLM(model_path=model_path),
        }
    if model_name not in dispatch:
        raise ValueError(f"Unknown model '{model_name}'. Choices: {list(dispatch)}")
    print(f"Loading model: {model_name} from {model_path}")
    return dispatch[model_name]()


def build_guess_query(row: dict) -> str:
    reason = row.get('image_reason', '')
    osm_results = row.get('osm', None)
    comment = row.get('comment', {})
    rag = row.get('retrieved_content', {})

    rag_formed = ''
    rag_threshold = 30
    for rag_key, rag_items in rag.items():
        if not rag_items:
            continue
        valid = [item for item in rag_items if item['distance'] <= rag_threshold]
        if not valid:
            continue
        clues = ' '.join(set(item['relevant_clue'] for item in valid))
        rag_formed += f'the relevant clues of {rag_key} in this image are: {clues}'

    comment_formed = ''
    for category, text in comment.items():
        if text:
            comment_formed += f'{category}: {text}\n'

    filtered_query = {
        k: [v for v in vals if v != 'None']
        for k, vals in row.get('genQuery', {}).items()
    }
    filtered_query = {k: v for k, v in filtered_query.items() if v}

    k_reason  = 1
    k_osm     = 1 if osm_results else 0
    k_rag     = 1 if rag_formed else 0
    k_comment = 1 if comment_formed else 0

    query = prompts.base_query + prompts.intro_query
    query += prompts.reason_query_template.format(reason=reason) * k_reason
    if k_osm:
        query += prompts.osm_query_template.format(
            filtered_Query=filtered_query, osm_results=osm_results)
    if k_comment:
        query += prompts.comment_query_template.format(comment_formed=comment_formed)
    if k_rag:
        query += prompts.rag_query_template.format(rag_formed=rag_formed)
    if k_reason:
        query += prompts.outro_query

    usage = {'reasoning': k_reason, 'osm': k_osm, 'rag': k_rag, 'comment': k_comment}
    return query, usage


def run_guess(s5_path, dataset_path, model, output_path, shard_id=0, num_shards=1):
    data = load_data(s5_path)
    if num_shards > 1:
        data = data[shard_id::num_shards]
        print(f'Shard {shard_id}/{num_shards}: processing {len(data)} samples')

    def _generate():
        for row in tqdm(data):
            image = os.path.join(dataset_path, 'images', row['ID'] + '.jpg')
            query, usage = build_guess_query(row)
            raw = model.base_inference(query, image)
            answer = parse_json(raw)
            print(f'raw: {raw!r}')
            print(f'parsed: {answer}  |  correct: {row["LAT"]}, {row["LON"]}')
            sys.stdout.flush()
            row['answer'] = answer
            row['usage'] = usage
            yield row

    dump_jsonl(_generate(), output_path)


def calculate_score(result_path):
    data = load_data(result_path)
    counts = [0, 0, 0, 0, 0]
    total_geoscore = 0.0
    total_dist = 0.0
    thresholds = [1, 25, 200, 750, 2500]
    n = 0
    for row in data:
        correct = [float(row['LAT']), float(row['LON'])]
        try:
            pred = [float(row['answer']['latitude']), float(row['answer']['longitude'])]
            dist = haversine_distance(pred, correct)
        except Exception:
            dist = 10000
        total_geoscore += Geoscore(dist)
        total_dist += dist
        for i, t in enumerate(thresholds):
            if dist <= t:
                counts[i] += 1
        n += 1
    if n == 0:
        print('No results found.')
        return
    score = [c / n for c in counts]
    print(f'\n=== Results: {result_path} ===')
    print(f'N = {n}')
    print(f'Accuracy @1km/@25km/@200km/@750km/@2500km: {[f"{s:.3f}" for s in score]}')
    print(f'Avg GeoScore: {total_geoscore / n:.2f}')
    print(f'Avg Distance: {total_dist / n:.2f} km')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--s5_path', type=str, required=True,
                   help='Path to results_s5.jsonl (merged or single shard)')
    p.add_argument('--dataset_path', type=str, required=True,
                   help='Dataset root containing images/ subdirectory')
    p.add_argument('--model', type=str, default='llama32vision',
                   choices=['llava', 'qwen', 'cpm', 'cpm_sft', 'llama32vision',
                            'internvl2', 'deepseek', 'falcon'])
    p.add_argument('--model_path', type=str, required=True,
                   help='Local path or HuggingFace model ID for the guesser model')
    p.add_argument('--ckpt_dir', type=str, default=None,
                   help='LoRA checkpoint directory (required for cpm_sft)')
    p.add_argument('--output', type=str, required=True,
                   help='Output JSONL path for stage-6 results')
    p.add_argument('--num_shards', type=int, default=1)
    p.add_argument('--shard_id', type=int, default=0)
    p.add_argument('--score_only', action='store_true',
                   help='Skip inference, just score an existing --output file')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    if not args.score_only:
        model = load_model(args.model, args.model_path, ckpt_dir=args.ckpt_dir)
        run_guess(args.s5_path, args.dataset_path, model,
                  args.output, args.shard_id, args.num_shards)

    calculate_score(args.output)
