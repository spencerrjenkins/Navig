# -*- coding: utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import warnings
warnings.filterwarnings("ignore", message=".*num_additional_image_tokens.*")
warnings.filterwarnings("ignore", message=".*trust_remote_code.*")
warnings.filterwarnings("ignore", message=".*Expanding inputs for image tokens.*")

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType,
    get_default_template_type, inference_stream
)
from swift.utils import seed_everything
from swift.tuners import Swift
import torch
import json
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM
import base64

from openai import OpenAI
from configuration import Config


def _fix_llava_next_processor(tokenizer):
    """Set patch_size and vision_feature_select_strategy on the LLaVA-NeXT processor
    to avoid incorrect image token expansion and the associated deprecation warning."""
    candidates = [tokenizer]
    if hasattr(tokenizer, 'processor'):
        candidates.append(tokenizer.processor)
    for obj in candidates:
        if hasattr(obj, 'patch_size') and obj.patch_size is None:
            obj.patch_size = 14
        if hasattr(obj, 'vision_feature_select_strategy') and obj.vision_feature_select_strategy is None:
            obj.vision_feature_select_strategy = 'default'


class LLaVA:

    # inference code of LLaVA model
    def __init__(self, model_path='vlms/llava/llava-v1.6-vicuna-7b-hf'):
        self.model_type = 'llava1_6-vicuna-7b-instruct'
        self.template_type = get_default_template_type(self.model_type)
        self.model_path = model_path
        self.model, self.tokenizer = get_model_tokenizer(self.model_type, torch.float16, model_id_or_path=self.model_path,
                                        model_kwargs={'device_map': 'auto'})
        self.model.generation_config.max_new_tokens = 256
        self.template = get_template(self.template_type, self.tokenizer)
        _fix_llava_next_processor(self.tokenizer)
        seed_everything(42)


    def base_inference(self, query, image_path = None):
        query = '<image>' + query
        if image_path and not isinstance(image_path,list):
            image_path = [image_path]
        response, _ = inference(self.model, self.template, query, images=image_path)
        return response


class LLaVA_sft:

    # inference code of LLaVA model
    def __init__(self, model_path='vlms/llava/llava-v1.6-vicuna-7b-hf', ckpt_dir="vlms/llava/checkpoint-534"):
        self.ckpt_dir = ckpt_dir
        self.model_type = 'llava1_6-vicuna-7b-instruct'
        self.template_type = get_default_template_type(self.model_type)
        self.model_path = model_path
        self.model, self.tokenizer = get_model_tokenizer(self.model_type, torch.float16, model_id_or_path=self.model_path,
                                        model_kwargs={'device_map': 'auto'})
        self.model.generation_config.max_new_tokens = 256
        self.template = get_template(self.template_type, self.tokenizer)
        _fix_llava_next_processor(self.tokenizer)
        seed_everything(42)
        self.model = Swift.from_pretrained(self.model, self.ckpt_dir, inference_mode=True)


    def base_inference(self, query, image_path):
        query = '<image>' + query
        image_path = [image_path]
        response, _ = inference(self.model, self.template, query, images=image_path)
        return response


class Qwen:

    # inference code of Qwen model
    def __init__(self, model_path='vlms/qwen/Qwen2-VL-7B-Instruct'):
        self.model_type = 'qwen2-vl-7b-instruct'
        self.template_type = get_default_template_type(self.model_type)
        self.model_path = model_path
        self.model, self.tokenizer = get_model_tokenizer(self.model_type, torch.float16, model_id_or_path=self.model_path,
                                        model_kwargs={'device_map': 'auto'})
        self.model.generation_config.max_new_tokens = 256
        self.template = get_template(self.template_type, self.tokenizer)
        seed_everything(42)


    def base_inference(self, query, image_path = None):
        query = '<image>' + query
        if image_path and not isinstance(image_path,list):
            image_path = [image_path]
        response, _ = inference(self.model, self.template, query, images=image_path)
        return response


class Qwen_sft:

    # inference code of Qwen model
    def __init__(self, model_path='vlms/qwen/Qwen2-VL-7B-Instruct', ckpt_dir="vlms/qwen/checkpoint-534"):
        self.ckpt_dir = ckpt_dir
        self.model_type = 'qwen2-vl-7b-instruct'
        self.template_type = get_default_template_type(self.model_type)
        self.model_path = model_path
        self.model, self.tokenizer = get_model_tokenizer(self.model_type, torch.float16, model_id_or_path=self.model_path,
                                        model_kwargs={'device_map': 'auto'})
        self.model.generation_config.max_new_tokens = 256
        self.template = get_template(self.template_type, self.tokenizer)
        seed_everything(42)
        self.model = Swift.from_pretrained(self.model, self.ckpt_dir, inference_mode=True)


    def base_inference(self, query, image_path):
        query = '<image>' + query
        image_path = [image_path]
        response, _ = inference(self.model, self.template, query, images=image_path)
        return response


class CPM:
    def __init__(self, model_path='vlms/cpm/MiniCPM-V-2_6'):
        self.model_type = 'minicpm-v-v2_6-chat'
        self.template_type = get_default_template_type(self.model_type)
        self.model_path = model_path
        self.model, self.tokenizer = get_model_tokenizer(self.model_type, torch.float16, model_id_or_path=self.model_path,
                                        model_kwargs={'device_map': 'auto'})
        self.model.generation_config.max_new_tokens = 256
        self.template = get_template(self.template_type, self.tokenizer)
        seed_everything(42)


    def base_inference(self, query, image_path = None):
        query = '<image>' + query
        if image_path and not isinstance(image_path,list):
            image_path = [image_path]

        response, _ = inference(self.model, self.template, query, images=image_path)
        return response



class CPM_sft:

    def __init__(self, model_path='vlms/cpm/MiniCPM-V-2_6', ckpt_dir="vlms/cpm/checkpoint-534"):
        self.ckpt_dir = ckpt_dir
        self.model_type = 'minicpm-v-v2_6-chat'
        self.template_type = get_default_template_type(self.model_type)
        self.model_path = model_path
        self.model, self.tokenizer = get_model_tokenizer(self.model_type, torch.float16, model_id_or_path=self.model_path,
                                        model_kwargs={'device_map': 'auto'})
        self.model.generation_config.max_new_tokens = 256
        self.template = get_template(self.template_type, self.tokenizer)
        seed_everything(42)
        self.model = Swift.from_pretrained(self.model, self.ckpt_dir, inference_mode=True)


    def base_inference(self, query, image_path):
        query = '<image>' + query
        image_path = [image_path]
        response, _ = inference(self.model, self.template, query, images=image_path)
        return response


# Vicuna system + turn template used by LLaVA-1.6-Vicuna
_LLAVA_VICUNA_PROMPT = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions. "
    "USER: <image>\n{query}\nASSISTANT:"
)


class LLaVA_vllm:
    """vLLM-accelerated LLaVA-NeXT base model inference.

    Replaces LLaVA for stages 4/5/6 when --use_vllm is set.
    Requires: pip install vllm==0.4.3  (last release with CUDA 11.8 support)
    """

    def __init__(self, model_path='vlms/llava/llava-v1.6-vicuna-7b-hf'):
        from vllm import LLM, SamplingParams
        self.llm = LLM(
            model=model_path,
            dtype="float16",
            max_model_len=4096,
        )
        self.sampling_params = SamplingParams(max_tokens=256, temperature=0)

    def base_inference(self, query, image_path=None):
        from PIL import Image as PILImage
        prompt = _LLAVA_VICUNA_PROMPT.format(query=query)
        if image_path:
            if isinstance(image_path, list):
                image_path = image_path[0]
            img = PILImage.open(image_path).convert("RGB")
            outputs = self.llm.generate(
                {"prompt": prompt, "multi_modal_data": {"image": img}},
                self.sampling_params,
            )
        else:
            outputs = self.llm.generate(prompt, self.sampling_params)
        return outputs[0].outputs[0].text


class Llama32Vision:
    """Llama-3.2-11B-Vision-Instruct via ms-Swift.

    Stronger reasoning than the 7B models already in NAVIG while still fitting
    comfortably in 48 GB VRAM (~22 GB at fp16).  Used as the drop-in guesser
    for the stage-6 swap experiment.

    Download:
        huggingface-cli download meta-llama/Llama-3.2-11B-Vision-Instruct \
            --local-dir /fs/nexus-scratch/$USER/llama-3.2-11b-vision-instruct
    """

    def __init__(self, model_path='vlms/llama32/llama-3.2-11b-vision-instruct'):
        self.model_type = 'llama3_2-11b-vision-instruct'
        self.template_type = get_default_template_type(self.model_type)
        self.model_path = model_path
        self.model, self.tokenizer = get_model_tokenizer(
            self.model_type, torch.float16, model_id_or_path=self.model_path,
            model_kwargs={'device_map': 'auto'},
        )
        self.model.generation_config.max_new_tokens = 256
        self.template = get_template(self.template_type, self.tokenizer)
        seed_everything(42)

    def base_inference(self, query, image_path=None):
        query = '<image>' + query
        if image_path and not isinstance(image_path, list):
            image_path = [image_path]
        response, _ = inference(self.model, self.template, query, images=image_path)
        return response


class InternVL2:
    """InternVL2-8B via ms-Swift.

    Particularly strong on text-rich and OCR tasks — a useful alternative for
    the stage-6 swap experiment when sign-reading quality is the bottleneck.

    Download:
        huggingface-cli download OpenGVLab/InternVL2-8B \
            --local-dir /fs/nexus-scratch/$USER/InternVL2-8B
    """

    def __init__(self, model_path='vlms/internvl2/InternVL2-8B'):
        self.model_type = 'internvl2-8b'
        self.template_type = get_default_template_type(self.model_type)
        self.model_path = model_path
        self.model, self.tokenizer = get_model_tokenizer(
            self.model_type, torch.bfloat16, model_id_or_path=self.model_path,
            model_kwargs={'device_map': 'auto'},
        )
        self.model.generation_config.max_new_tokens = 256
        self.template = get_template(self.template_type, self.tokenizer)
        seed_everything(42)

    def base_inference(self, query, image_path=None):
        query = '<image>' + query
        if image_path and not isinstance(image_path, list):
            image_path = [image_path]
        response, _ = inference(self.model, self.template, query, images=image_path)
        return response


class LLaVA_sft_vllm:
    """vLLM-accelerated LLaVA-NeXT with PEFT LoRA adapter.

    Replaces LLaVA_sft for stage 1 when --use_vllm is set.
    The adapter at ckpt_dir must contain adapter_config.json + adapter_model.safetensors
    in HuggingFace PEFT format (Swift checkpoints satisfy this).
    """

    def __init__(self, model_path='vlms/llava/llava-v1.6-vicuna-7b-hf',
                 ckpt_dir='vlms/NAVIG/llava1_6-vicuna-7b-instruct'):
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        self.llm = LLM(
            model=model_path,
            dtype="float16",
            max_model_len=4096,
            enable_lora=True,
            max_lora_rank=16,
        )
        self.sampling_params = SamplingParams(max_tokens=256, temperature=0)
        self.lora_request = LoRARequest("navig_sft", 1, ckpt_dir)

    def base_inference(self, query, image_path):
        from PIL import Image as PILImage
        if isinstance(image_path, list):
            image_path = image_path[0]
        prompt = _LLAVA_VICUNA_PROMPT.format(query=query)
        img = PILImage.open(image_path).convert("RGB")
        outputs = self.llm.generate(
            {"prompt": prompt, "multi_modal_data": {"image": img}},
            self.sampling_params,
            lora_request=self.lora_request,
        )
        return outputs[0].outputs[0].text
