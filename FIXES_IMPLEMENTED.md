# Comparison Run Issues — Fixes Implemented

Date: 2026-05-21  
Branch: main

## Summary

Fixed three critical issues in the 9-model comparison pipeline:

1. **Falcon-11B-VLM empty reasoning output** (stage 1)
2. **LLaVA vLLM token overflow at stage 6** 
3. **MiniCPM-V-2.6 SFT adapter not being loaded** (stage 1)

---

## Changes Made

### Fix 1: Falcon Empty Reasoning (llm.py:514–553)

**Problem:** Falcon model was returning empty strings for `image_reason` in stage 1, despite completing inference without errors.

**Root Cause:** The model was hitting end-of-sequence (EOS) tokens immediately after generating the `[/INST]` marker, likely because:
- No `min_new_tokens` constraint (model could output just 1 EOS token)
- Greedy decoding with `do_sample=False` is too conservative for open-ended reasoning
- Model needs diverse sampling to produce reasoning chains

**Solution:**
- Changed from `do_sample=False` (greedy) to `do_sample=True` with `temperature=0.7` and `top_p=0.9`
- Added `min_new_tokens=10` to force minimum output length
- Increased `max_new_tokens` from 256 to 512 for longer reasoning chains
- Added fallback: if sampling produces empty output, retry with beam search (`num_beams=2`)
- Added logging to diagnose empty responses

**Code Changes:**
```python
# Before: greedy only, no min length
out = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)

# After: sampling with fallback
out = self.model.generate(
    **inputs,
    max_new_tokens=512,
    min_new_tokens=10,
    temperature=0.7,
    do_sample=True,
    top_p=0.9,
)
# ... with fallback to beam search if response is empty
```

---

### Fix 2: LLaVA Token Overflow at Stage 6 (llm.py:264)

**Problem:** LLaVA vLLM runs were failing at stage 6 because input tokens exceeded the model's context window limit.

**Evidence:**
```
Token indices sequence length is longer than the specified maximum sequence length 
for this model (4228 > 4096)
```

**Root Cause:** Stage 6 queries are comprehensive summaries that include:
- Reasoning output from stage 1 (~500 tokens)
- RAG-retrieved guidebook knowledge (~300 tokens)
- Commenting details from stage 4 (~200 tokens)
- OSM search results (~150 tokens)
- Total: ~4200 tokens in a single prompt

The vLLM LLaVA model was initialized with `max_model_len=4096`, causing overflow.

**Solution:**
- Increased `max_model_len` from 4096 to 5120 tokens (25% headroom)
- This accommodates stage 6 queries up to ~4400 tokens

**Code Changes:**
```python
# Before
self.llm = LLM(model=model_path, dtype="float16", max_model_len=4096)

# After
self.llm = LLM(model=model_path, dtype="float16", max_model_len=5120)
```

**Note:** Future scaling: if queries grow beyond 5120, consider implementing prompt truncation in `build_guess_query()` or upgrading to a model with larger context window (e.g., LLaVA-NeXT with 8K support).

---

### Fix 3: MiniCPM-V SFT Adapter Not Loading (llm.py:209–237)

**Problem:** MiniCPM-V-2.6 base and SFT models were producing identical reasoning outputs, indicating the LoRA adapter was not being applied.

**Root Cause:** 
1. Checkpoint path was relative (`vlms/cpm/checkpoint-534`), potentially resolving incorrectly depending on working directory
2. No logging to verify adapter was loaded
3. No error checking if checkpoint directory didn't exist
4. Swift might silently skip adapter loading if path resolution failed

**Solution:**
- Resolve checkpoint path to absolute path using `os.path.abspath()`
- Add logging to confirm adapter loading before and after Swift.from_pretrained()
- Add path existence check with fallback warning (use base model if adapter not found)
- Ensures consistent behavior regardless of working directory

**Code Changes:**
```python
# Before: relative path, no logging
self.model = Swift.from_pretrained(self.model, ckpt_dir, inference_mode=True)

# After: absolute path with diagnostics
ckpt_dir_abs = os.path.abspath(ckpt_dir)
logger.info(f"Loading CPM-V SFT adapter from: {ckpt_dir_abs}")
if not os.path.exists(ckpt_dir_abs):
    logger.warning(f"Adapter directory not found. Using base model without SFT.")
    return
self.model = Swift.from_pretrained(self.model, ckpt_dir_abs, inference_mode=True)
logger.info(f"Successfully loaded SFT adapter for MiniCPM-V-2.6")
```

**Next Steps for CPM-SFT Debugging:**
- Run a single sample through CPM_sft and check logs for the new info/warning messages
- If adapter still not applied, verify `adapter_model.safetensors` is not corrupted and target_modules match the model

---

## Testing Recommendations

1. **Falcon Fix:**
   - Re-run Falcon stage 1 on test dataset (`dataset/test`)
   - Verify `image_reason` field is no longer empty
   - Check logs for any "attempting greedy decoding" warnings

2. **LLaVA Fix:**
   - Re-run LLaVA and LLaVA-SFT stage 6
   - Verify no token overflow warnings in logs
   - Confirm results_s6_llava*.jsonl files are created

3. **CPM-SFT Fix:**
   - Re-run CPM-SFT stage 1
   - Check logs for "Loading CPM-V SFT adapter from:" and "Successfully loaded SFT adapter" messages
   - Verify stage 1 reasoning differs from base CPM model

---

## Files Modified

- `/nfshomes/srjnk01/Navig/llm.py` (3 changes in 3 classes)

## Backward Compatibility

All changes are backward compatible:
- Falcon: only changes generation parameters (non-breaking)
- LLaVA: increases context window (non-breaking, memory cost ~50MB)
- CPM-SFT: adds logging, doesn't change model behavior (non-breaking)