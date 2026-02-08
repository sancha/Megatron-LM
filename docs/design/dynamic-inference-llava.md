# Design: DynamicInferenceEngine + LLaVA Integration

## Problem Statement

The `DynamicInferenceEngine` provides high-throughput serving for decoder-only LLMs
with dynamic batching, block-level KV cache (paged attention), CUDA graphs, and
chunked prefill. However, it only supports pure text models today. The LLaVA
multimodal model (vision encoder + projector + language model) has its own inference
path via `VLMInferenceWrapper` and `StaticInferenceEngine`, which lacks dynamic
batching, continuous batching, and paged attention.

This document lays out a plan to make the dynamic inference engine work with LLaVA
models, bringing all of its performance benefits to multimodal inference.

## Architecture Overview

### Current Dynamic Engine (text-only)

```
add_request(text) → tokenize → DynamicInferenceRequest
                                      ↓
                            DynamicInferenceContext
                          (block KV cache, token tracking)
                                      ↓
                         step_modern() loop:
                           _dynamic_step_context_init()
                             → input_ids, position_ids
                           _dynamic_step_forward_logits(input_ids, position_ids)
                             → GPTInferenceWrapper._forward()
                               → model(tokens, position_ids, attention_mask=None)
                             → logits
                           sample → append token → repeat
```

Key files:
- `megatron/core/inference/engines/dynamic_engine.py` — `DynamicInferenceEngine`
- `megatron/core/inference/contexts/dynamic_context.py` — `DynamicInferenceContext`
- `megatron/core/inference/model_inference_wrappers/gpt/gpt_inference_wrapper.py`
- `megatron/core/inference/text_generation_controllers/text_generation_controller.py`
- `examples/inference/gpt/gpt_dynamic_inference.py` — entry point

### Current LLaVA Inference (static)

```
LLaVAModel.forward(images, input_ids, position_ids, ...):
  1. vision_model(images) → [num_tiles, img_seq_len, h_vision]
  2. vision_projection(emb) → [img_seq_len, num_tiles, h_language]
  3. language_model.embedding(input_ids) → [text_seq_len, batch, h_language]
  4. _preprocess_data() merges image+text embeddings:
       - Each -200 token expands to img_seq_len * tiles image embeddings
       - Output: [combined_seq_len, batch, h_language]
  5. language_model(decoder_input=combined) → logits
```

Key files:
- `megatron/core/models/multimodal/llava_model.py` — `LLaVAModel`
- `megatron/core/inference/model_inference_wrappers/multimodal/vlm_inference_wrapper.py`
- `megatron/core/inference/text_generation_controllers/vlm_text_generation_controller.py`
- `examples/multimodal/run_text_generation.py` — entry point

## Core Challenges

1. **Token expansion**: A single `-200` placeholder token expands into 576+
   image embedding vectors during prefill. The dynamic context's token accounting
   (block allocation, token count tracking, position IDs) assumes a 1:1 mapping
   between input tokens and KV cache positions.

2. **No image data path**: `add_request()` only accepts text/tokens.
   `_dynamic_step_forward_logits()` only passes `(input_ids, position_ids)` to
   the model wrapper. No mechanism exists to carry images through the pipeline.

3. **Per-request variable expansion**: Each request in a batch may have different
   numbers of images and tiles, so the "expanded" sequence length varies per
   request — while the dynamic context manages them in a flat token buffer.

4. **CUDA graph incompatibility**: Prefill with image embedding insertion has
   variable shapes that conflict with CUDA graph requirements.

5. **Chunked prefill splitting**: Chunk boundaries must not split an image
   embedding region.

6. **Eviction/re-scheduling**: Evicted requests lose their KV cache. On re-add,
   image embeddings must be re-inserted during re-prefill.

## Strategy: Pre-Expand, Embed-Outside

The central insight is: **compute vision embeddings before tokens enter the
dynamic context, and pre-expand the prompt token sequence to its true
post-vision-merge length.** This makes the expansion invisible to the dynamic
context, which simply sees a longer prompt of tokens. During the forward pass,
the wrapper replaces the dummy image token positions with the pre-computed
embeddings.

```
add_request(text + images):
  1. Tokenize text
  2. Run images through vision encoder + projection → image_embeddings
  3. Expand prompt tokens: replace each -200 with (img_seq_len * tiles) dummy tokens
  4. Store image_embeddings on the request
  5. Pass expanded tokens to DynamicInferenceContext

Forward pass (VLMDynamicInferenceWrapper):
  Prefill:
    1. Receive flat input_ids from dynamic context
    2. Embed text tokens via language_model.embedding()
    3. Replace dummy image positions with pre-computed image_embeddings
    4. Call language_model(decoder_input=combined_embeddings)
  Decode:
    1. Standard GPT path (all image embeddings are in KV cache)
```

## Implementation Plan

### Phase 1: Request Data Model

**Goal**: Allow `DynamicInferenceRequest` and `add_request()` to carry image data
and compute the expanded token length upfront.

#### 1A. Extend `DynamicInferenceRequest`

File: `megatron/core/inference/inference_request.py`

Add optional fields (defaulting to `None` so text-only requests are unaffected):

```python
@dataclass(kw_only=True)
class DynamicInferenceRequest:
    ...
    # Multimodal fields (None for text-only requests)
    images: Optional[torch.Tensor] = None             # [num_tiles, C, H, W]
    num_image_tiles: Optional[torch.Tensor] = None     # [num_images]
    image_embeddings: Optional[torch.Tensor] = None    # [img_seq_len, total_tiles, h]
    image_token_positions: Optional[List[Tuple[int, int]]] = None  # (start, end) in expanded seq
```

#### 1B. Extend `DynamicInferenceEngine.add_request()`

File: `megatron/core/inference/engines/dynamic_engine.py`

Add optional `images` and `num_image_tiles` parameters. When images are provided:

1. Count the number of `-200` placeholder tokens in the prompt.
2. For each placeholder, compute the expanded length (`img_seq_len * tiles`).
3. Build a synthetic "expanded" prompt token tensor where placeholder positions
   are filled with dummy token IDs (e.g., 0).
4. Record the `(start, end)` position ranges of each image in the expanded sequence.
5. Store images and metadata on the request object.

The expanded prompt tokens become `request.prompt_tokens` and
`request.remaining_prompt_tokens`, so the dynamic context sees the correct total
length for block allocation.

#### 1C. Vision Embedding Pre-computation

Compute vision embeddings at request-addition time (not during the batched forward
pass). This runs the LLaVA model's `vision_model` and `vision_projection` modules:

```python
with torch.inference_mode():
    emb = model.vision_model(images)          # [num_tiles, img_seq_len, h_vision]
    if drop_class_token:
        emb = emb[:, class_token_len:, :]
    if pixel_shuffle:
        emb = pixel_shuffle(emb)
    emb = emb.permute(1, 0, 2).contiguous()   # [img_seq_len, num_tiles, h_vision]
    emb = model.vision_projection(emb)         # [img_seq_len, num_tiles, h_language]
```

Store the result as `request.image_embeddings`. This is a relatively cheap one-time
cost per request and avoids the complexity of batching vision encoding across
requests with different image counts.

### Phase 2: VLM Dynamic Inference Wrapper

**Goal**: Create a new model wrapper that handles multimodal forward passes,
inserting pre-computed image embeddings at the right positions in the flat token
buffer.

#### 2A. Create `VLMDynamicInferenceWrapper`

New file: `megatron/core/inference/model_inference_wrappers/multimodal/vlm_dynamic_inference_wrapper.py`

Extends `GPTInferenceWrapper` and overrides `_forward()`.

**Prefill path** (when active requests have `image_embeddings`):

1. Receive `inference_input = {"tokens": input_ids, "position_ids": ..., "attention_mask": None}`.
2. The `input_ids` tensor is flat (shape `[1, total_tokens]`) containing expanded
   tokens from all active requests concatenated.
3. Embed all tokens via `self.model.language_model.embedding(input_ids, position_ids)`.
4. For each active request that has `image_embeddings`:
   - Use `image_token_positions` to identify the slice in the flat sequence.
   - Replace the language embeddings at those positions with the pre-computed
     `image_embeddings`.
5. Call `self.model.language_model(input_ids=None, decoder_input=combined_embeddings, ...)`.
6. Mark image embeddings as consumed. Free them from the request to reclaim memory.

**Decode path** (no image embeddings):

Fall back to the standard GPT path: pass `input_ids` and `position_ids` directly
to the model. The LLaVA model detects `use_inference_kv_cache = True` and skips
vision processing.

#### 2B. Accessing per-request data from the wrapper

The wrapper needs to know which tokens in the flat buffer belong to which
request, and whether each request has image data. The dynamic context provides:

- `token_to_request_idx`: maps each token position to its request index.
- Request metadata accessible via the context's request data structures.
- The wrapper can maintain a reference to the engine's request dict (or the
  image embedding store).

**Preferred approach**: The wrapper accesses image embeddings directly from the
`DynamicInferenceRequest` objects via the context. This keeps
`TextGenerationController._dynamic_step_forward_logits()` unchanged — it
continues to pass only `(input_ids, position_ids)`.

### Phase 3: KV Cache Block Allocation

**Goal**: Verify that the block allocator correctly handles expanded sequences.

Since we pre-expand the prompt tokens (Phase 1B), the `DynamicInferenceContext`
methods that matter already work correctly:

- `check_availability()` uses `req.remaining_prompt_length` (= len of expanded tokens) ✓
- `add_request()` allocates blocks based on `chunk_length` (from expanded tokens) ✓
- `request_output_lengths` includes the expanded prompt length ✓
- `max_sequence_length` must be set to `max_expanded_prompt_length + max_gen_length` ✓

**Action**: Primarily verification. The entry point script must compute
`max_sequence_length` using expanded prompt lengths rather than raw text lengths.

### Phase 4: Chunked Prefill Compatibility

**Goal**: Ensure chunked prefill does not split image embedding regions.

#### Option A: Disable chunked prefill for VLM requests (recommended for v1)

The simplest approach: when a request has images, ensure it is never chunked. This
can be done by setting a flag on the request or by having the scheduling logic
check for image data.

This is acceptable because:
- Image requests typically have moderate prompt lengths (text + expanded images).
- The vision encoder is already a one-time cost.
- Correctness is more important than maximum interleaving for the first version.

#### Option B: Image-boundary-aware chunking (future optimization)

Track image embedding boundaries per request. When computing chunk boundaries,
snap to the nearest image boundary. This requires:
- A per-request `image_boundaries: List[int]` field.
- Modification to the scheduling logic in `dynamic_engine.py` to respect boundaries.

### Phase 5: CUDA Graph Compatibility

**Goal**: Ensure CUDA graphs work correctly with VLM requests.

#### Decode-only CUDA graphs: No changes needed

The decode step for VLM requests is identical to GPT decode — one token per
request, no images. Existing CUDA graphs work as-is.

#### Prefill CUDA graphs: Disable for VLM requests (recommended for v1)

For prefill steps involving image embedding insertion, disable CUDA graphs. The
dynamic context already supports `use_cuda_graphs_for_non_decode_steps=False`.
When VLM requests are present in a prefill batch, skip graph replay and run
eagerly.

Future optimization: pre-capture separate CUDA graphs for the VLM prefill path
with fixed image embedding sizes.

### Phase 6: Entry Point Script

**Goal**: Create the VLM dynamic inference entry point.

New file: `examples/inference/vlm/vlm_dynamic_inference.py`

This script mirrors `examples/inference/gpt/gpt_dynamic_inference.py` but:

1. Uses a LLaVA model provider (from `pretrain_vlm.py` or `examples/multimodal/model.py`).
2. Creates a `VLMDynamicInferenceWrapper` instead of `GPTInferenceWrapper`.
3. Loads requests with images (from file paths or a JSON manifest).
4. Preprocesses images using `examples/multimodal/image_processing.py` utilities
   (resize, tile, normalize).
5. Calls `engine.add_request(id, prompt, params, images=images, num_image_tiles=tiles)`.
6. Computes `max_sequence_length` with expanded prompt lengths.

### Phase 7: REST API Extension

**Goal**: Allow the server endpoints to accept multimodal requests.

#### 7A. Extend `/v1/chat/completions` endpoint

File: `megatron/core/inference/text_generation_server/dynamic_text_gen_server/endpoints/chat_completions.py`

Add support for OpenAI-compatible multimodal messages:

```json
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "What is in this image?"},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ]
  }]
}
```

Parse image data from base64 or URL, preprocess, and pass to the engine.

#### 7B. Extend coordinator protocol

File: `megatron/core/inference/data_parallel_inference_coordinator.py`

Add an optional `images` field to the `SUBMIT_REQUEST` message:

```python
[Headers.SUBMIT_REQUEST, request_id, prompt, sampling_params, images_bytes]
```

The serialized image tensor (or `None`) is passed alongside the token data.

### Phase 8: Edge Cases

#### Text-only requests in a VLM setup

When the VLM is loaded but a request has no images, the system should work
identically to the GPT path. The wrapper detects `image_embeddings is None` and
falls back to the standard `_forward()`. The LLaVA model already handles this
(empty image embeddings tensor).

#### Multiple images per request

LLaVA supports multiple `-200` placeholders. The expansion handles each image
independently: each placeholder expands to `img_seq_len * tiles_for_that_image`.
The `image_token_positions` list has one `(start, end)` pair per image.

#### Eviction and re-scheduling

When a request is evicted (memory pressure), its KV cache is freed. On re-add,
the request must be re-prefilled including the image embeddings. Since image
embeddings are stored on the request object, they survive eviction. The
expanded prompt tokens are also preserved. On re-prefill, the wrapper
re-inserts the image embeddings at the correct positions.

**Important**: Do NOT free image embeddings after the first prefill if eviction
is possible. Only free them when the request completes.

#### Pipeline parallelism

The vision encoder lives on PP stage 0. Since we pre-compute vision embeddings
at request-addition time (also on stage 0), the PP communication path only
needs to handle the combined language embeddings, which is the same as the
existing GPT PP path.

## Implementation Order

| Order | Phase | Description | Effort |
|-------|-------|-------------|--------|
| 1 | Phase 1 | Request data model + expanded tokens | Medium |
| 2 | Phase 1C | Vision embedding pre-computation | Medium |
| 3 | Phase 2 | VLM dynamic inference wrapper | High |
| 4 | Phase 3 | KV cache verification | Low |
| 5 | Phase 6 | Entry point script | Medium |
| 6 | Phase 4 | Chunked prefill (disable for v1) | Low |
| 7 | Phase 5 | CUDA graphs (disable prefill graphs for v1) | Low |
| 8 | Phase 7 | REST API extension | Medium |
| 9 | Phase 8 | Edge cases (eviction, PP, multi-image) | High |

## Files to Create

- `megatron/core/inference/model_inference_wrappers/multimodal/vlm_dynamic_inference_wrapper.py`
- `examples/inference/vlm/vlm_dynamic_inference.py`

## Files to Modify

- `megatron/core/inference/inference_request.py` — add image fields to `DynamicInferenceRequest`
- `megatron/core/inference/engines/dynamic_engine.py` — extend `add_request()` for images
- `megatron/core/inference/text_generation_server/dynamic_text_gen_server/endpoints/chat_completions.py` — multimodal messages
- `megatron/core/inference/data_parallel_inference_coordinator.py` — image data in coordinator protocol

## Files Unchanged (verification only)

- `megatron/core/inference/contexts/dynamic_context.py` — works with expanded tokens as-is
- `megatron/core/inference/text_generation_controllers/text_generation_controller.py` — `_dynamic_step_forward_logits` unchanged (wrapper handles image insertion internally)
- `megatron/core/models/multimodal/llava_model.py` — model unchanged, wrapper drives it differently
