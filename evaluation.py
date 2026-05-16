from tqdm import tqdm
import json
import re
from utils import load_data, dump_jsonl, parse_json, haversine_distance, search_place_with_retry, PatchImages
from utils import retrieve_similar_images, _parse_osm_candidates
from llm import LLaVA, Qwen, LLaVA_sft, Qwen_sft, CPM, CPM_sft, LLaVA_vllm, LLaVA_sft_vllm, DeepSeekVL, FalconVLM, Llama32Vision
import os
import argparse
import numpy as np
import sys
from PIL import Image
import prompts


def _parse_coord(value):
    """Parse a coordinate value that may be a plain float, degree-notation string
    ('8.6836° N'), or hedged string ('Approximately 75° W').
    Raises ValueError for unparseable values like 'Unknown' or JSON null (None)."""
    if value is None:
        raise ValueError("Unparseable coordinate: None")
    s = str(value).strip()
    if s.lower() in ('unknown', 'n/a', '', 'nan', 'null', 'none', 'infinity', 'undefined'):
        raise ValueError(f"Unparseable coordinate: {s!r}")
    s = re.sub(r'(?i)approximately\s*', '', s).strip()
    m = re.match(r'^(-?\d+(?:\.\d+)?)\s*°?\s*([NSEWnsew])?$', s)
    if m:
        val = float(m.group(1))
        if (m.group(2) or '').upper() in ('S', 'W'):
            val = -val
        return val
    return float(s)
"""
Define the following parameters for evaluation:
    dataset: name of the dataset, could be Gws15k, im2gps3k, OpenWorld, and yfcc4k.
    model: the model used in the pipeline, could be llava or qwen.
    output_path: the output path.
"""

def Geoscore(distance):
    return 5000 * np.exp(-distance / 1492.7)

class Evaluator:

    def __init__(self, dataset_path, model, output_path, crop_box_treshold, crop_text_treshold, model_path, ckpt_dir, use_vllm=False, shard_id=0, num_shards=1):
        self.dataset_path = dataset_path
        self.model_type = model
        self.output_path = output_path
        self.base_model = None  # loaded lazily after stage 1 to avoid holding two 7B models in GPU memory simultaneously
        self.crop_box_treshold = crop_box_treshold
        self.crop_text_treshold = crop_text_treshold
        self.ckpt_dir = ckpt_dir
        self.model_path = model_path
        self.use_vllm = use_vllm
        self.shard_id = shard_id
        self.num_shards = num_shards



    def getReasoning(self):
        print("///////////The 1st stage: Reasoning///////////")
        # load dataset
        data = load_data(dataset_path+"/meta.jsonl") # it's dataset path because this is the first step
        if self.num_shards > 1:
            data = data[self.shard_id::self.num_shards]
            print(f"Shard {self.shard_id}/{self.num_shards}: processing {len(data)} samples")
        # load model
        # Note: the SFT model always uses Swift inference regardless of --use_vllm,
        # because vLLM's LoRA+multimodal support for LLaVA-NeXT is not stable.
        # vLLM speedup applies to the base model in stages 4-6.
        if self.model_type == "llava":
            reasoning_model = LLaVA_sft(model_path=self.model_path, ckpt_dir=self.ckpt_dir)
        elif self.model_type == "qwen":
            reasoning_model = Qwen_sft(model_path=self.model_path, ckpt_dir=self.ckpt_dir)
        elif self.model_type == "llama32vision":
            reasoning_model = Llama32Vision(model_path=self.model_path)
        elif self.model_type == "deepseek":
            reasoning_model = DeepSeekVL(model_path=self.model_path)
        elif self.model_type == "falcon":
            reasoning_model = FalconVLM(model_path=self.model_path)
        else:
            reasoning_model = CPM_sft(model_path=self.model_path, ckpt_dir=self.ckpt_dir)
        for row in tqdm(data):
            image = f"{dataset_path}/images/{row['ID']}.jpg"
            query = prompts.reasoning_prompt
            response = reasoning_model.base_inference(query, image)
            row["image_reason"] = response
            yield row



        
    def getGrounding(self):
        print("///////////The 2th stage: Grounding///////////")
        data = load_data(output_path+"/results_s1.jsonl")
        ground = PatchImages(['road sign', 'house', 'building sign'])
        patch_dir = os.path.join(dataset_path,'patchesV2')
        
        if not os.path.exists(patch_dir):
            os.mkdir(patch_dir)
        for row in tqdm(data):
            image_path = os.path.join(dataset_path,'images',row['ID'] + '.jpg')
            patchResult = ground(image_path,BOX_TRESHOLD=self.crop_box_treshold,TEXT_TRESHOLD=self.crop_text_treshold)
            tmp = {}
            for type, cropped_ls in patchResult.items() :
               tmp[type] = []
               for i, cropped_img in enumerate(cropped_ls):
                try:
                    cropped_img = Image.fromarray(cropped_img)
                    save_path = os.path.join(patch_dir, f"{row['ID']}_{type}_{i}.jpg")
                    cropped_img.save(save_path)
                    tmp[type].append(save_path)
                except:
                    continue
            row['crop']=tmp
            yield row
        

    

    def getRAG(self):
        print("///////////The 3th stage: Retriving with Grounding Images///////////")
        data = load_data(output_path+"/results_s2.jsonl")
        for row in tqdm(data):
            retrieved_dict = {}
            crop_dict = row["crop"]
            items = crop_dict.keys()
            for item in items:
                images = crop_dict[item]
                if images:
                    sim_images, sim_texts, distances = retrieve_similar_images(images[0], threshold=40)
                    if sim_images:
                        retrieved_dict[item] = [
                            {
                                "similar_image": sim_images[i],
                                "relevant_clue": sim_texts[i],
                                "distance": distances[i]
                            } for i in range(len(sim_images))
                        ]
                    else:
                        retrieved_dict[item] = []
                else:
                    retrieved_dict[item] = []
            
            row["retrieved_content"] = retrieved_dict
            yield row




    def getComment(self):
        print("///////////The 4th stage: Commenting with Grounding Images///////////")
        data = load_data(output_path + "/results_s3.jsonl")
        
        for row in tqdm(data):
            commented_dict = {}
            crop_dict = row["crop"]
            items = crop_dict.keys()
            for item in items:
                if item == 'house':
                    k = 3
                else:
                    k = len(crop_dict[item])
                commented_dict[item] = ""
                query = prompts.comment_gen_template.format(item = item)
                images = crop_dict[item][:k]
                if images:
                    for image in images:
                        response = self.base_model.base_inference(query,image)
                        commented_dict[item] += response + "\t"
                else:
                    commented_dict[item] = ""
            
            row["comment"] = commented_dict
            yield row

    def getOSM(self):
        print("///////////The 5rd stage: Search OCR///////////")
        data = load_data(output_path+"/results_s4.jsonl")
        prompt = prompts.osm_gen
        for row in tqdm(data):
            row['genQuery'] = {}
            row['osm'] = None
            for category in row["crop"].keys():
                row['genQuery'][category] = []
                images = row["crop"][category]
                for image in images:
                    query = self.base_model.base_inference(prompt, image)
                    row['genQuery'][category].append(query)
                    candidates = _parse_osm_candidates(query)
                    if not candidates:
                        continue
                    response = search_place_with_retry(query, top_k=3)
                    if response is not None:
                        if row['osm'] is None:
                            row['osm'] = response
                        else:
                            row['osm'].extend(response)
            yield row
    

    def guessCoordinates(self, only_ids=None):
        print("///////////The 6th stage: Guessing the Coordinates///////////")
        data = load_data(output_path+"/results_s5.jsonl")
        if only_ids is not None:
            data = [row for row in data if row['ID'] in only_ids]
        '''
        create a query. this query forms like:
            base_query: a base query for models, which can be used for directly test model performance.
            intro_query: an intro query to introduce models other information that it can refer to.
            reason_query: the reasoning provided by our model.
            osm_query: searching results of the osm.
            rag_query: form rag results as query, with a distance threshold.
            outro_query: outro. if not including the extra information, the outro is not needed.
        '''
        base_query = prompts.base_query
        for row in tqdm(data):
            image = f"{dataset_path}/images/{row['ID']}.jpg"
            reason = row.get("image_reason","")
            osm_results = row.get("osm", None)
            comment = row.get('comment', {})
            rag = row.get("retrieved_content", {})
            rag_formed = ""
            rag_threshold = 30
            rag_keys = rag.keys()
            for rag_key in rag_keys:
                if not rag[rag_key]:
                    continue
                valid_results = [item for item in rag[rag_key] if item["distance"] <= rag_threshold]
                if not valid_results:
                    continue
                unique_clues = list(set(item["relevant_clue"] for item in valid_results))
                clues = ' '.join([item for item in unique_clues])
                rag_formed += f"the relevant clues of {rag_key} in this image are: {clues}"
            
            comment_formed=""
            for category in comment.keys():
                if not comment[category]:
                    continue
                comment_formed += f"{category}: {comment[category]} \n"

            filtered_Query = {key: [v for v in value if v != 'None'] for key, value in row.get('genQuery',{}).items()}
            filtered_Query = {key: value for key, value in filtered_Query.items() if value}

            # define the queries
            intro_query = prompts.intro_query
            reason_query = prompts.reason_query_template.format(reason = reason)
            
            rag_query = prompts.rag_query_template.format(rag_formed = rag_formed)
            comment_query = prompts.comment_query_template.format(comment_formed = comment_formed)
            osm_query = prompts.osm_query_template.format(filtered_Query = filtered_Query, osm_results = osm_results)
            outro_query = prompts.outro_query

            # decide whether to append the query or not, using keys
            k_intro = 1
            k_reason = 1
            k_osm = 1 if osm_results else 0
            k_rag = 1 if rag_formed else 0
            k_comment = 1 if comment_formed else 0
            k_outro = 1 if k_reason else 0
            # form the queries
            usage = {"reasoning": k_reason, "osm": k_osm, "rag": k_rag, "comment": k_comment}
            row["usage"] = usage
            query = base_query + intro_query * k_intro + reason_query * k_reason + osm_query * k_osm + comment_query * k_comment + rag_query * k_rag + outro_query * k_outro
            answer = self.base_model.base_inference(query, image) 
            #print(f"model response {answer}")
            answer = parse_json(answer)
            #print(f"parser response {answer}")
            #print("correct answer:", row["LAT"], row["LON"])
            sys.stdout.flush()
            row["answer"] = answer
            yield row


    def _load_base_model(self):
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        if self.model_type == 'qwen':
            self.base_model = Qwen(model_path=self.model_path)
        elif self.model_type == 'llava':
            if self.use_vllm:
                self.base_model = LLaVA_vllm(model_path=self.model_path)
            else:
                self.base_model = LLaVA(model_path=self.model_path)
        elif self.model_type == 'llama32vision':
            self.base_model = Llama32Vision(model_path=self.model_path)
        elif self.model_type == 'deepseek':
            self.base_model = DeepSeekVL(model_path=self.model_path)
        elif self.model_type == 'falcon':
            self.base_model = FalconVLM(model_path=self.model_path)
        else:
            self.base_model = CPM(model_path=self.model_path)

    def forward(self):
        os.makedirs(output_path, exist_ok=True)
        # Stage 1 (uses reasoning_model internally, not self.base_model):
        o_file = f"{output_path}/results_s1.jsonl"
        dump_jsonl(self.getReasoning(), o_file)

        # Load base model now that reasoning_model has been freed
        self._load_base_model()

        # Stage 2:
        o_file = f"{output_path}/results_s2.jsonl"
        dump_jsonl(self.getGrounding(), o_file)

        # Stage 3:
        o_file = f"{output_path}/results_s3.jsonl"
        dump_jsonl(self.getRAG(), o_file)

        # Stage 4:
        o_file = f"{output_path}/results_s4.jsonl"
        dump_jsonl(self.getComment(), o_file)

        # Stage 5:
        o_file = f"{output_path}/results_s5.jsonl"
        dump_jsonl(self.getOSM(), o_file)

    def guess_forward(self):
        o_file = f"{output_path}/{results_fileName}"
        dump_jsonl(self.guessCoordinates(), o_file)

    def retry_guess_forward(self):
        """Re-run stage 6 only for rows where answer=None (JSON parse failed entirely).
        Rows with answer set but 'Unknown' coordinates are valid model outputs and are
        NOT retried — the model expressed genuine uncertainty. Patches file in-place."""
        existing_path = f"{output_path}/{results_fileName}"
        if not os.path.exists(existing_path):
            print(f"No existing results at {existing_path}; running full stage 6.")
            self.guess_forward()
            return
        existing = load_data(existing_path)

        # Only retry rows where JSON parsing failed completely.  Rows where the
        # model returned {"latitude": "Unknown", ...} are valid outputs — retrying
        # them wastes GPU time and produces the same result.
        def _is_failed(row):
            return row.get('answer') is None

        failed_ids = {row['ID'] for row in existing if _is_failed(row)}
        if not failed_ids:
            print(f"No failed rows found in {existing_path}. Nothing to retry.")
            return
        print(f"Retrying {len(failed_ids)} failed rows out of {len(existing)} total...")
        retry_map = {row['ID']: row for row in self.guessCoordinates(only_ids=failed_ids)}
        merged = [retry_map.get(row['ID'], row) for row in existing]
        dump_jsonl(merged, existing_path)
        still_failed = sum(1 for row in merged if _is_failed(row))
        print(f"Updated {len(retry_map)} rows. Still failing after retry: {still_failed}.")


    def calculate_score(self):
        data = load_data(f"{output_path}/{results_fileName}")
        counts = [0, 0, 0, 0, 0]
        total_points = 0
        Distance = 0
        thresholds = [1, 25, 200, 750, 2500]
        for row in data:
            correct_answer = [float(row["LAT"]), float(row["LON"])]
            try:
                guessed_answer = [_parse_coord(row["answer"]["latitude"]), _parse_coord(row["answer"]["longitude"])]
                distance = haversine_distance(guessed_answer, correct_answer)
            except:
                guessed_answer = [0, 0]
                distance = 10000
            points = Geoscore(distance)
            total_points += points
            Distance += distance
            # row["distance"] = distance
            # print(guessed_answer, correct_answer, distance)
            for i, t in enumerate(thresholds):
                if distance <= t:
                    counts[i] += 1
        total_num = len(data)
        score = [count / total_num for count in counts]
        # print(f"Five LeveL: Street-> Continent{score}")
        # print(f"Avg.Geoscore is {total_points/total_num}")
        # print(f"Avg.distance is {Distance/total_num}")

    def calculate_score_cc(self):
        data = load_data(f"{output_path}/{results_fileName}")
        counts = [0, 0]
        for row in data:
            try:
                correct_country = row["country"]
                correct_city = row["city"]
                predicted_country = row["answer"]["country"]
                predicted_city = row["answer"]["city"]
                if correct_country in predicted_country:
                    counts[0] += 1
                if correct_city in predicted_city:
                    counts[1] += 1
            except:
                continue
        total_num = len(data)
        score = [count / total_num for count in counts]
        # print(score)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default="im2gps3k",help="Please input valid dataset path,only for im2gps3k, yfcc4k, and Gws5k")
    parser.add_argument('--model', type=str, default='qwen', choices=["qwen", "llava", "cpm", "deepseek", "falcon", "llama32vision"])
    parser.add_argument('--reasoning_path',  type=str, default ='.')
    parser.add_argument('--results_file_Name', type=str, default = 'Final_results.jsonl')
    parser.add_argument('--crop_box_treshold', type=float, default = 0.65)
    parser.add_argument('--crop_text_treshold', type=float, default = 0.55)
    parser.add_argument('--model_path', type=str, default = 'vlms/qwen/Qwen2-VL-7B-Instruct')
    parser.add_argument('--ckpt_dir', type=str, default = 'vlms/qwen/checkpoint-534')
    parser.add_argument('--use_vllm', action='store_true', help="Use vLLM for faster LLaVA inference")
    parser.add_argument('--num_shards', type=int, default=1, help="Total number of parallel shards")
    parser.add_argument('--shard_id', type=int, default=0, help="Which shard this job processes (0-indexed)")
    parser.add_argument('--stage6_only', action='store_true', help="Skip stages 1-5; run only stage 6 on existing results_s5.jsonl")
    parser.add_argument('--retry_failed', action='store_true', help="Re-run stage 6 only for rows with answer=None in the existing output; patches file in-place (implies --stage6_only)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(args)
    dataset_path = args.dataset_path
    model = args.model
    output_path = args.reasoning_path
    results_fileName = args.results_file_Name
    crop_box_treshold = args.crop_box_treshold
    crop_text_treshold = args.crop_text_treshold
    model_path = args.model_path
    ckpt_dir = args.ckpt_dir
    use_vllm = args.use_vllm
    num_shards = args.num_shards
    shard_id = args.shard_id

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    evaluator = Evaluator(dataset_path, model, output_path, crop_box_treshold, crop_text_treshold, model_path, ckpt_dir, use_vllm=use_vllm, shard_id=shard_id, num_shards=num_shards)
    if args.stage6_only or args.retry_failed:
        evaluator._load_base_model()
    else:
        evaluator.forward()

    if args.retry_failed:
        evaluator.retry_guess_forward()
    else:
        evaluator.guess_forward()
    evaluator.calculate_score()
    evaluator.calculate_score_cc()


