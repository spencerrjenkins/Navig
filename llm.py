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
        # The SFT checkpoint ships a generation_config.json with top_k=1 and
        # temperature=0.01, which collapses the sampling distribution and causes
        # the model to emit EOS immediately on the free-form reasoning prompt.
        # Reset to greedy decoding with generous length before any inference.
        self.model.generation_config.do_sample = False
        self.model.generation_config.top_k = 0
        self.model.generation_config.top_p = 1.0
        self.model.generation_config.temperature = 1.0
        self.model.generation_config.max_new_tokens = 512


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


class DeepSeekVL:
    """DeepSeek-VL-7B-Chat via ms-Swift.

    NOTE: Used zero-shot (no NAVIG SFT) as a stage-6 guesser only.
    ms-Swift natively supports deepseek-vl-7b-chat and handles the <image>
    token mapping through its template system, same as LLaVA and Qwen.

    Download:
        huggingface-cli download deepseek-ai/deepseek-vl-7b-chat \
            --local-dir /fs/nexus-scratch/$USER/deepseek-vl-7b-chat
    """

    def __init__(self, model_path='vlms/deepseek/deepseek-vl-7b-chat'):
        self.model_type = 'deepseek-vl-7b-chat'
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


class FalconVLM:
    """Falcon-11B-VLM via HuggingFace Transformers (ms-Swift does not support this model).

    Falcon-11B-VLM is built on LLaVA-NeXT, so it requires LlavaNextProcessor
    and LlavaNextForConditionalGeneration.  The prompt must contain the literal
    <image> token inside [INST]…[/INST] tags; using AutoProcessor or passing
    images as a list causes out-of-bounds embedding index errors.

    Download:
        huggingface-cli download tiiuae/falcon-11B-vlm \
            --local-dir /fs/nexus-scratch/$USER/falcon-11B-vlm
    """

    def __init__(self, model_path='vlms/falcon/falcon-11B-vlm'):
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        from transformers.models.falcon.modeling_falcon import FalconForCausalLM
        self.processor = LlavaNextProcessor.from_pretrained(model_path)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map='auto')
        # The <image> special token sits at index len(tokenizer)-1 = 65024,
        # but the saved embedding matrix only has 65024 rows (0-65023).
        # Resize to cover the full vocabulary so the image token lookup
        # doesn't trigger an out-of-bounds CUDA assertion.
        old_vocab_size = self.model.get_input_embeddings().weight.shape[0]
        self.model.resize_token_embeddings(len(self.processor.tokenizer))
        # Initialize the newly added row(s) to the mean of existing embeddings.
        # A zero-initialized image-token embedding causes the model to generate
        # EOS immediately (the visual context appears blank to the LM).
        with torch.no_grad():
            in_emb = self.model.get_input_embeddings()
            in_emb.weight.data[old_vocab_size:] = (
                in_emb.weight.data[:old_vocab_size].mean(dim=0, keepdim=True)
            )
            out_emb = self.model.get_output_embeddings()
            if out_emb is not None and out_emb.weight.shape[0] > old_vocab_size:
                out_emb.weight.data[old_vocab_size:] = (
                    out_emb.weight.data[:old_vocab_size].mean(dim=0, keepdim=True)
                )
        self.model.generation_config.max_new_tokens = 256
        # Newer LlavaNext passes num_logits_to_keep to the inner LM, but the
        # installed FalconForCausalLM predates that argument — patch it out.
        _orig_forward = FalconForCausalLM.forward
        def _forward_compat(self, *args, **kwargs):
            kwargs.pop('num_logits_to_keep', None)
            return _orig_forward(self, *args, **kwargs)
        FalconForCausalLM.forward = _forward_compat

    def base_inference(self, query, image_path=None):
        from PIL import Image as PILImage
        img = None
        if image_path:
            p = image_path[0] if isinstance(image_path, list) else image_path
            img = PILImage.open(p).convert('RGB')

        prompt = f'[INST] <image>\n{query} [/INST]' if img else f'[INST] {query} [/INST]'

        device = next(self.model.parameters()).device
        inputs = self.processor(text=prompt, images=img, return_tensors='pt').to(device)
        out = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        new_tokens = out[:, inputs['input_ids'].shape[1]:]
        return self.processor.decode(new_tokens[0], skip_special_tokens=True)


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
