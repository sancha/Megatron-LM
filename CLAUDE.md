# Dynamic Inference for LLaVA — Reference Guide

This document summarizes the key files and concepts for the **DynamicInferenceEngine + LLaVA integration**. Use it as a starting point for debugging or extending the VLM dynamic inference path.

## Architecture Overview

The integration uses a **"pre-expand, embed-outside"** strategy:
1. Vision embeddings are computed at request-addition time (before entering the engine loop).
2. The prompt token sequence is expanded to its true post-vision-merge length, replacing each `-200` image placeholder with `img_seq_len * num_tiles` dummy tokens.
3. During prefill, the `VLMDynamicInferenceWrapper._forward()` splices pre-computed image embeddings into the token embedding buffer before calling the language model.
4. Decode steps fall through to the standard GPT path (no image data involved).

## Files Created/Modified

### Created

| File | Purpose |
|------|---------|
| `megatron/core/inference/model_inference_wrappers/multimodal/vlm_dynamic_inference_wrapper.py` | Core wrapper: image data registry, `compute_image_embeddings()`, `_forward()` with embedding splice |
| `megatron/core/inference/multimodal/image_preprocessing.py` | `ImageTransform`, tiling, normalization (copied from `examples/multimodal/` so core can access it) |
| `examples/inference/vlm/vlm_dynamic_inference.py` | Entry point for VLM dynamic inference |
| `examples/inference/vlm/compare_static_vs_dynamic.py` | Comparison script: runs same image+prompt through static and dynamic engines, compares greedy output |
| `docs/design/dynamic-inference-llava.md` | Design document with full architecture and implementation plan |

### Modified

| File | What changed |
|------|-------------|
| `megatron/core/inference/inference_request.py` | Added `image_embeddings`, `image_token_positions`, `has_images` fields to `DynamicInferenceRequest`; carried them forward in `checkpoint()` |
| `megatron/core/inference/engines/dynamic_engine.py` | Extended `add_request()` with image params; added `_expand_prompt_for_images()`; added wrapper register/unregister calls; guarded chunked prefill for VLM requests |

## Key Reference Files

### Dynamic Inference Engine

- **`megatron/core/inference/engines/dynamic_engine.py`**
  - `add_request()` / `_add_request()` — request admission, prompt expansion, wrapper registration
  - `step_modern()` — main inference loop step (prefill + decode scheduling)
  - `schedule_non_chunked_prefill()` / `schedule_chunked_prefill()` — prefill scheduling (VLM uses non-chunked)
  - `post_process_requests()` — finished request cleanup, calls `unregister_image_data()`

- **`megatron/core/inference/contexts/dynamic_context.py`**
  - `DynamicInferenceContext` — KV cache block allocation, `memory_buffer`, `block_allocator`
  - `add_request()` — allocates KV blocks for the expanded prompt length
  - `current_input_and_position_ids()` — produces the flat token buffer and position IDs passed to the wrapper
  - `append_key_value_cache()` — scatter-writes K/V into paged blocks
  - `request_ids`, `request_query_lengths` — used by VLM wrapper to map flat buffer offsets

### Model Wrappers

- **`megatron/core/inference/model_inference_wrappers/multimodal/vlm_dynamic_inference_wrapper.py`**
  - `register_image_data()` / `unregister_image_data()` — lifecycle management
  - `compute_image_embeddings()` — vision_model → drop class token → pixel shuffle → permute → vision_projection → tile tagging
  - `_forward()` — the critical method: embeds tokens, splices image embeddings into flat buffer, calls language model with `decoder_input`

- **`megatron/core/inference/model_inference_wrappers/gpt/gpt_inference_wrapper.py`**
  - Base `_forward()` that VLM wrapper falls back to for decode steps
  - `prep_inference_input()` — builds the `{"tokens", "position_ids", "attention_mask"}` dict

- **`megatron/core/inference/model_inference_wrappers/multimodal/vlm_inference_wrapper.py`**
  - The **static** VLM wrapper (for comparison/debugging). Uses `run_one_forward_step()` with `image_tokens_count` tracking

- **`megatron/core/inference/model_inference_wrappers/abstract_model_inference_wrapper.py`**
  - Base class with pipeline parallel forward methods

### Text Generation Controllers

- **`megatron/core/inference/text_generation_controllers/text_generation_controller.py`**
  - `_dynamic_step_forward_logits()` — passes `{"tokens", "position_ids", "attention_mask"}` to wrapper
  - Used by DynamicInferenceEngine for both GPT and VLM paths

- **`megatron/core/inference/text_generation_controllers/vlm_text_generation_controller.py`**
  - Used only by the **static** VLM path. Calls `VLMInferenceWrapper.prep_inference_input()` with image data

### LLaVA Model

- **`megatron/core/models/multimodal/llava_model.py`**
  - `LLaVAModel.forward()` — lines ~850-900: vision encoding, `_preprocess_data()` for image/text merging
  - `_preprocess_data()` — the static path's image-text merge logic (dynamic path bypasses this)
  - `_apply_tile_tagging()` — adds tile boundary tags to image embeddings
  - `IMAGE_TOKEN = "<image>"` with token ID `-200`
  - `pixel_shuffle()` — downsamples vision features

- **`megatron/core/models/vision/clip_vit_model.py`**
  - `get_num_image_embeddings()` — computes `img_seq_len` per tile based on patch size, class token, pixel shuffle, tile tags

### Image Preprocessing

- **`megatron/core/inference/multimodal/image_preprocessing.py`** (core copy)
- **`examples/multimodal/image_processing.py`** (original)
  - `ImageTransform.__call__()` — dispatches to `dynamic_preprocess()` when tiling is enabled
  - `dynamic_preprocess()` — finds closest aspect ratio, splits image into tiles
  - `find_closest_aspect_ratio()` — aspect ratio matching algorithm
  - `_build_transform()` — per-vision-model normalization (CLIP, SigLIP, InternViT, RADIO, etc.)

### Inference Requests

- **`megatron/core/inference/inference_request.py`**
  - `DynamicInferenceRequest` — has `image_embeddings`, `image_token_positions`, `has_images`
  - `DynamicInferenceRequestRecord.checkpoint()` — must carry multimodal fields forward for eviction recovery
  - `VLMInferenceRequest` — used only by the static path

### Example Entry Points

- **`examples/inference/gpt/gpt_dynamic_inference.py`** — reference for text-only dynamic inference setup
- **`examples/multimodal/run_text_generation.py`** — reference for static VLM inference
- **`examples/inference/vlm/vlm_dynamic_inference.py`** — VLM dynamic inference entry point
- **`examples/inference/vlm/compare_static_vs_dynamic.py`** — correctness comparison script

## Debugging Guide

### If static and dynamic outputs diverge

1. **Check token expansion**: Print `_expand_prompt_for_images()` output. Verify expanded length matches what the static path computes in `_preprocess_data()`. The formula: `expanded_len = text_tokens + sum(tiles * img_seq_len) - num_placeholders`.

2. **Check image embedding values**: Compare `VLMDynamicInferenceWrapper.compute_image_embeddings()` output against `LLaVAModel.forward()` vision encoding. They should be identical tensors.

3. **Check flat buffer splice positions**: In `VLMDynamicInferenceWrapper._forward()`, print `flat_offset`, `flat_start`, `flat_end` for each image region. Verify they align with where the expanded dummy tokens are in the flat buffer.

4. **Check the language embedding layer**: The dummy tokens (value 0) at image positions get embedded by the language model's embedding layer, then overwritten. Verify `language_embeddings[flat_start:flat_end]` matches the pre-computed `img_emb_flat[:length]` after splice.

5. **Check position IDs**: The dynamic context assigns sequential position IDs. Verify they match what the static path assigns after image expansion.

6. **Check KV cache**: Both paths should produce identical K/V entries. The dynamic path writes via `append_key_value_cache()` with block-level paging; the static path writes via contiguous cache. Values should match for the same positions.

### Common pitfalls

- **Chunked prefill + VLM**: Chunked prefill must be disabled for VLM requests because chunk boundaries can split image embedding regions. The guard is in `dynamic_engine.py`'s scheduling logic.
- **Non-decode CUDA graphs + VLM**: Disabled because image embedding insertion has variable shapes per request.
- **Eviction recovery**: If a VLM request is evicted and re-prefilled, the image data must still be in `_vlm_image_data`. The `checkpoint()` method carries `image_embeddings` and `image_token_positions` forward, and `unregister_image_data()` is only called when the request truly finishes.
- **Pipeline parallelism**: `compute_image_embeddings()` only works on PP stage 0 (where the vision model lives). Other stages receive the combined embeddings via normal PP send/recv.
- **Image token ID**: The placeholder token is `-200` (defined in `llava_model.py` as `IMAGE_TOKEN_INDEX`). The tokenizer encodes `<image>` to this value.

### Useful commands

```bash
# Run VLM dynamic inference
python examples/inference/vlm/vlm_dynamic_inference.py \
    --load <checkpoint_path> \
    --input-image-path <image_dir> \
    --num-tokens-to-generate 128 \
    --use-tiling --max-num-tiles 4

# Compare static vs dynamic output
python examples/inference/vlm/compare_static_vs_dynamic.py \
    --load <checkpoint_path> \
    --input-image-path <image_file> \
    --num-tokens-to-generate 64 \
    --use-tiling --max-num-tiles 4
```
