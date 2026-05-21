# -*- coding: utf-8 -*-
"""VLM wrappers for the NAVIG pipeline.

Every class exposes a single unified method::

    base_inference(query: str, image_path: str | None) -> str

All models use fp16 and greedy decoding by default.  The ``load_model``
factory at the bottom of this file provides a single entry point used by
both ``pipeline/evaluation.py`` and ``pipeline/guess_only.py``.
"""

import warnings
warnings.filterwarnings("ignore", message=".*num_additional_image_tokens.*")
warnings.filterwarnings("ignore", message=".*trust_remote_code.*")
warnings.filterwarnings("ignore", message=".*Expanding inputs for image tokens.*")

import torch

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType,
    get_default_template_type, inference_stream,
)
from swift.utils import seed_everything
from swift.tuners import Swift
from transformers import AutoProcessor, AutoModelForCausalLM
import base64

from openai import OpenAI
from configuration import Config


def _fix_llava_next_processor(tokenizer) -> None:
    """Patch deprecated fields on the LLaVA-NeXT processor to suppress warnings
    and prevent incorrect image token expansion."""
    for obj in [tokenizer, getattr(tokenizer, "processor", None)]:
        if obj is None:
            continue
        if hasattr(obj, "patch_size") and obj.patch_size is None:
            obj.patch_size = 14
        if (hasattr(obj, "vision_feature_select_strategy")
                and obj.vision_feature_select_strategy is None):
            obj.vision_feature_select_strategy = "default"


# ── LLaVA-1.6-Vicuna-7B ───────────────────────────────────────────────────────

class LLaVA:
    def __init__(self, model_path: str = "vlms/llava/llava-v1.6-vicuna-7b-hf"):
        self.model_type = "llava1_6-vicuna-7b-instruct"
        self.template_type = get_default_template_type(self.model_type)
        seed_everything(42)
        self.model, self.tokenizer = get_model_tokenizer(
            self.model_type, torch.float16,
            model_id_or_path=model_path, model_kwargs={"device_map": "auto"},
        )
        self.model.generation_config.max_new_tokens = 256
        self.template = get_template(self.template_type, self.tokenizer)
        _fix_llava_next_processor(self.tokenizer)

    def base_inference(self, query: str, image_path=None) -> str:
        query = "<image>" + query
        if image_path and not isinstance(image_path, list):
            image_path = [image_path]
        response, _ = inference(self.model, self.template, query, images=image_path)
        return response


class LLaVA_sft:
    def __init__(
        self,
        model_path: str = "vlms/llava/llava-v1.6-vicuna-7b-hf",
        ckpt_dir: str = "vlms/llava/checkpoint-534",
    ):
        self.model_type = "llava1_6-vicuna-7b-instruct"
        self.template_type = get_default_template_type(self.model_type)
        seed_everything(42)
        self.model, self.tokenizer = get_model_tokenizer(
            self.model_type, torch.float16,
            model_id_or_path=model_path, model_kwargs={"device_map": "auto"},
        )
        self.model.generation_config.max_new_tokens = 256
        self.template = get_template(self.template_type, self.tokenizer)
        _fix_llava_next_processor(self.tokenizer)
        self.model = Swift.from_pretrained(self.model, ckpt_dir, inference_mode=True)

    def base_inference(self, query: str, image_path: str) -> str:
        response, _ = inference(
            self.model, self.template, "<image>" + query, images=[image_path]
        )
        return response


# ── Qwen2-VL-7B-Instruct ──────────────────────────────────────────────────────

class Qwen:
    def __init__(self, model_path: str = "vlms/qwen/Qwen2-VL-7B-Instruct"):
        self.model_type = "qwen2-vl-7b-instruct"
        self.template_type = get_default_template_type(self.model_type)
        seed_everything(42)
        self.model, self.tokenizer = get_model_tokenizer(
            self.model_type, torch.float16,
            model_id_or_path=model_path, model_kwargs={"device_map": "auto"},
        )
        self.model.generation_config.max_new_tokens = 256
        self.template = get_template(self.template_type, self.tokenizer)

    def base_inference(self, query: str, image_path=None) -> str:
        query = "<image>" + query
        if image_path and not isinstance(image_path, list):
            image_path = [image_path]
        response, _ = inference(self.model, self.template, query, images=image_path)
        return response


class Qwen_sft:
    def __init__(
        self,
        model_path: str = "vlms/qwen/Qwen2-VL-7B-Instruct",
        ckpt_dir: str = "vlms/qwen/checkpoint-534",
    ):
        self.model_type = "qwen2-vl-7b-instruct"
        self.template_type = get_default_template_type(self.model_type)
        seed_everything(42)
        self.model, self.tokenizer = get_model_tokenizer(
            self.model_type, torch.float16,
            model_id_or_path=model_path, model_kwargs={"device_map": "auto"},
        )
        self.model.generation_config.max_new_tokens = 256
        self.template = get_template(self.template_type, self.tokenizer)
        self.model = Swift.from_pretrained(self.model, ckpt_dir, inference_mode=True)
        # The SFT checkpoint ships a generation_config.json with top_k=1 and
        # temperature=0.01, which collapses the sampling distribution and causes
        # the model to emit EOS immediately on the free-form reasoning prompt.
        # Reset to greedy decoding with generous length before any inference.
        self.model.generation_config.do_sample = False
        self.model.generation_config.top_k = 0
        self.model.generation_config.top_p = 1.0
        self.model.generation_config.temperature = 1.0
        self.model.generation_config.max_new_tokens = 512

    def base_inference(self, query: str, image_path: str) -> str:
        response, _ = inference(
            self.model, self.template, "<image>" + query, images=[image_path]
        )
        return response


# ── MiniCPM-V-2.6 ─────────────────────────────────────────────────────────────

class CPM:
    def __init__(self, model_path: str = "vlms/cpm/MiniCPM-V-2_6"):
        self.model_type = "minicpm-v-v2_6-chat"
        self.template_type = get_default_template_type(self.model_type)
        seed_everything(42)
        self.model, self.tokenizer = get_model_tokenizer(
            self.model_type, torch.float16,
            model_id_or_path=model_path, model_kwargs={"device_map": "auto"},
        )
        self.model.generation_config.max_new_tokens = 256
        self.template = get_template(self.template_type, self.tokenizer)

    def base_inference(self, query: str, image_path=None) -> str:
        query = "<image>" + query
        if image_path and not isinstance(image_path, list):
            image_path = [image_path]
        response, _ = inference(self.model, self.template, query, images=image_path)
        return response


class CPM_sft:
    def __init__(
        self,
        model_path: str = "vlms/cpm/MiniCPM-V-2_6",
        ckpt_dir: str = "vlms/cpm/checkpoint-534",
    ):
        self.model_type = "minicpm-v-v2_6-chat"
        self.template_type = get_default_template_type(self.model_type)
        seed_everything(42)
        self.model, self.tokenizer = get_model_tokenizer(
            self.model_type, torch.float16,
            model_id_or_path=model_path, model_kwargs={"device_map": "auto"},
        )
        self.model.generation_config.max_new_tokens = 256
        self.template = get_template(self.template_type, self.tokenizer)
        self.model = Swift.from_pretrained(self.model, ckpt_dir, inference_mode=True)

    def base_inference(self, query: str, image_path: str) -> str:
        response, _ = inference(
            self.model, self.template, "<image>" + query, images=[image_path]
        )
        return response


# ── vLLM-accelerated LLaVA (base model, stages 4–6) ──────────────────────────

# Vicuna system + turn template used by LLaVA-1.6-Vicuna.
_LLAVA_VICUNA_PROMPT = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions. "
    "USER: <image>\n{query}\nASSISTANT:"
)


class LLaVA_vllm:
    """vLLM-accelerated LLaVA-NeXT base model for stages 4/5/6 (--use_vllm)."""

    def __init__(self, model_path: str = "vlms/llava/llava-v1.6-vicuna-7b-hf"):
        from vllm import LLM, SamplingParams
        seed_everything(42)
        self.llm = LLM(model=model_path, dtype="float16", max_model_len=4096)
        self.sampling_params = SamplingParams(max_tokens=256, temperature=0)

    def base_inference(self, query: str, image_path=None) -> str:
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


# ── Experimental: vLLM + LoRA for stage 1 ────────────────────────────────────
# NOTE: vLLM's LoRA + multimodal support for LLaVA-NeXT is not stable.
# This class is retained for experimentation but is NOT used by default.

class LLaVA_sft_vllm:
    """EXPERIMENTAL — vLLM-accelerated LLaVA-NeXT + PEFT LoRA for stage 1."""

    def __init__(
        self,
        model_path: str = "vlms/llava/llava-v1.6-vicuna-7b-hf",
        ckpt_dir: str = "vlms/NAVIG/llava1_6-vicuna-7b-instruct",
    ):
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        seed_everything(42)
        self.llm = LLM(
            model=model_path, dtype="float16", max_model_len=4096,
            enable_lora=True, max_lora_rank=16,
        )
        self.sampling_params = SamplingParams(max_tokens=256, temperature=0)
        self.lora_request = LoRARequest("navig_sft", 1, ckpt_dir)

    def base_inference(self, query: str, image_path: str) -> str:
        from PIL import Image as PILImage
        if isinstance(image_path, list):
            image_path = image_path[0]
        img = PILImage.open(image_path).convert("RGB")
        outputs = self.llm.generate(
            {"prompt": _LLAVA_VICUNA_PROMPT.format(query=query),
             "multi_modal_data": {"image": img}},
            self.sampling_params,
            lora_request=self.lora_request,
        )
        return outputs[0].outputs[0].text


# ── Llama-3.2-11B-Vision-Instruct ─────────────────────────────────────────────

class Llama32Vision:
    """Llama-3.2-11B-Vision-Instruct via ms-Swift (stage-6 swap experiment).

    Download::

        huggingface-cli download meta-llama/Llama-3.2-11B-Vision-Instruct \\
            --local-dir /fs/nexus-scratch/$USER/llama-3.2-11b-vision-instruct
    """

    def __init__(self, model_path: str = "vlms/llama32/llama-3.2-11b-vision-instruct"):
        self.model_type = "llama3_2-11b-vision-instruct"
        self.template_type = get_default_template_type(self.model_type)
        seed_everything(42)
        self.model, self.tokenizer = get_model_tokenizer(
            self.model_type, torch.float16,
            model_id_or_path=model_path, model_kwargs={"device_map": "auto"},
        )
        self.model.generation_config.max_new_tokens = 256
        self.template = get_template(self.template_type, self.tokenizer)

    def base_inference(self, query: str, image_path=None) -> str:
        query = "<image>" + query
        if image_path and not isinstance(image_path, list):
            image_path = [image_path]
        response, _ = inference(self.model, self.template, query, images=image_path)
        return response


# ── InternVL2-8B ──────────────────────────────────────────────────────────────

class InternVL2:
    """InternVL2-8B via ms-Swift (strong OCR; stage-6 swap experiment).

    Download::

        huggingface-cli download OpenGVLab/InternVL2-8B \\
            --local-dir /fs/nexus-scratch/$USER/InternVL2-8B
    """

    def __init__(self, model_path: str = "vlms/internvl2/InternVL2-8B"):
        self.model_type = "internvl2-8b"
        self.template_type = get_default_template_type(self.model_type)
        seed_everything(42)
        self.model, self.tokenizer = get_model_tokenizer(
            self.model_type, torch.bfloat16,
            model_id_or_path=model_path, model_kwargs={"device_map": "auto"},
        )
        self.model.generation_config.max_new_tokens = 256
        self.template = get_template(self.template_type, self.tokenizer)

    def base_inference(self, query: str, image_path=None) -> str:
        query = "<image>" + query
        if image_path and not isinstance(image_path, list):
            image_path = [image_path]
        response, _ = inference(self.model, self.template, query, images=image_path)
        return response


# ── DeepSeek-VL-7B-Chat ───────────────────────────────────────────────────────

class DeepSeekVL:
    """DeepSeek-VL-7B-Chat via ms-Swift (zero-shot stage-6 guesser).

    Download::

        huggingface-cli download deepseek-ai/deepseek-vl-7b-chat \\
            --local-dir /fs/nexus-scratch/$USER/deepseek-vl-7b-chat
    """

    def __init__(self, model_path: str = "vlms/deepseek/deepseek-vl-7b-chat"):
        self.model_type = "deepseek-vl-7b-chat"
        self.template_type = get_default_template_type(self.model_type)
        seed_everything(42)
        self.model, self.tokenizer = get_model_tokenizer(
            self.model_type, torch.float16,
            model_id_or_path=model_path, model_kwargs={"device_map": "auto"},
        )
        self.model.generation_config.max_new_tokens = 256
        self.template = get_template(self.template_type, self.tokenizer)

    def base_inference(self, query: str, image_path=None) -> str:
        query = "<image>" + query
        if image_path and not isinstance(image_path, list):
            image_path = [image_path]
        response, _ = inference(self.model, self.template, query, images=image_path)
        return response


# ── Falcon-11B-VLM ────────────────────────────────────────────────────────────

class FalconVLM:
    """Falcon-11B-VLM via HuggingFace Transformers (ms-Swift does not support it).

    Built on LLaVA-NeXT architecture.  Requires custom embedding resize to
    cover the <image> special token index which sits outside the saved matrix.

    Download::

        huggingface-cli download tiiuae/falcon-11B-vlm \\
            --local-dir /fs/nexus-scratch/$USER/falcon-11B-vlm
    """

    def __init__(self, model_path: str = "vlms/falcon/falcon-11B-vlm"):
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        from transformers.models.falcon.modeling_falcon import FalconForCausalLM

        seed_everything(42)
        self.processor = LlavaNextProcessor.from_pretrained(model_path)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )
        # The <image> token sits at index len(tokenizer)-1 = 65024,
        # but the saved embedding matrix only has 65024 rows (0–65023).
        # Resize and initialise the new row to the mean of existing embeddings
        # so the image token embedding is not zero (zero → immediate EOS).
        old_vocab_size = self.model.get_input_embeddings().weight.shape[0]
        self.model.resize_token_embeddings(len(self.processor.tokenizer))
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
            kwargs.pop("num_logits_to_keep", None)
            return _orig_forward(self, *args, **kwargs)

        FalconForCausalLM.forward = _forward_compat

    def base_inference(self, query: str, image_path=None) -> str:
        from PIL import Image as PILImage

        img = None
        if image_path:
            p = image_path[0] if isinstance(image_path, list) else image_path
            img = PILImage.open(p).convert("RGB")

        prompt = f"[INST] <image>\n{query} [/INST]" if img else f"[INST] {query} [/INST]"
        device = next(self.model.parameters()).device
        inputs = self.processor(text=prompt, images=img, return_tensors="pt").to(device)
        out = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.decode(new_tokens[0], skip_special_tokens=True)


# ── Shared model factory ───────────────────────────────────────────────────────

def load_model(
    model_name: str,
    model_path: str,
    ckpt_dir: str | None = None,
    use_vllm: bool = False,
):
    """Instantiate and return a VLM by name.

    *model_name* must be one of: llava, qwen, cpm, cpm_sft, llama32vision,
    internvl2, deepseek, falcon.

    ``ckpt_dir`` is required for SFT variants (cpm_sft).
    ``use_vllm`` enables vLLM acceleration for the llava base model.
    """
    if model_name == "llava":
        return (
            LLaVA_vllm(model_path=model_path)
            if use_vllm
            else LLaVA(model_path=model_path)
        )
    if model_name == "qwen":
        return Qwen(model_path=model_path)
    if model_name == "cpm":
        return CPM(model_path=model_path)
    if model_name == "cpm_sft":
        if not ckpt_dir:
            raise ValueError("--ckpt_dir is required for cpm_sft")
        return CPM_sft(model_path=model_path, ckpt_dir=ckpt_dir)
    if model_name == "llama32vision":
        return Llama32Vision(model_path=model_path)
    if model_name == "internvl2":
        return InternVL2(model_path=model_path)
    if model_name == "deepseek":
        return DeepSeekVL(model_path=model_path)
    if model_name == "falcon":
        return FalconVLM(model_path=model_path)
    raise ValueError(
        f"Unknown model {model_name!r}. "
        f"Choices: llava, qwen, cpm, cpm_sft, llama32vision, internvl2, deepseek, falcon"
    )


def load_sft_model(model_name: str, model_path: str, ckpt_dir: str):
    """Instantiate the SFT (LoRA-finetuned) reasoning model for stage 1."""
    if not ckpt_dir:
        # No checkpoint provided → use zero-shot base model for stage-1 reasoning.
        return load_model(model_name, model_path)
    if model_name == "llava":
        return LLaVA_sft(model_path=model_path, ckpt_dir=ckpt_dir)
    if model_name == "qwen":
        return Qwen_sft(model_path=model_path, ckpt_dir=ckpt_dir)
    if model_name == "cpm":
        return CPM_sft(model_path=model_path, ckpt_dir=ckpt_dir)
    # Non-SFT models fall back to their zero-shot base for stage 1.
    return load_model(model_name, model_path)
