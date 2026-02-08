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

## Image Preprocessing Pipeline

Understanding the full pipeline from raw image to embedding is critical because
the dynamic engine must run preprocessing before `add_request()`, and the
preprocessing determines `num_tiles` which determines the expanded sequence length.

### Existing code location

The preprocessing code lives in `examples/multimodal/image_processing.py` — NOT in
`megatron/core/`. This is a problem for the REST API server (which lives in
`megatron/core/inference/`).

### Step-by-step pipeline: raw image to embeddings

```
Step 1: Image Preprocessing (CPU, before add_request)
─────────────────────────────────────────────────────
  Input: PIL Image + model config (vision_model_type, img_h, img_w, use_tiling,
         max_num_tiles, use_thumbnail)

  ImageTransform.__call__()  [examples/multimodal/image_processing.py:72]
    │
    ├─ If use_tiling=True:
    │    dynamic_preprocess()  [line 88]
    │      1. Compute aspect ratio of input image
    │      2. Enumerate valid tile grids: (1,1), (1,2), (2,1), (2,2), ...
    │         up to max_num_tiles total tiles
    │      3. find_closest_aspect_ratio() picks best grid  [line 31]
    │      4. Resize image to (grid_w * img_h, grid_h * img_w)
    │      5. Crop into grid_w * grid_h tiles, each (img_h, img_w)
    │      6. If use_thumbnail: append a (img_h, img_w) thumbnail
    │    Apply _build_transform() to each tile
    │
    ├─ If use_tiling=False:
    │    Apply _build_transform() to the single image
    │
    └─ _build_transform()  [line 131]
         Per-vision-model normalization:
           clip:       CLIP mean/std, bicubic resize to (img_h, img_w)
           siglip:     [0.5, 0.5, 0.5] mean/std, bicubic resize
           internvit:  ImageNet mean/std, bicubic resize
           radio:      CLIP mean/std, bicubic resize
           radio-g:    RADIO-G mean/std, bicubic resize

  Output: List[Tensor], each [C=3, img_h, img_w]
  Stack → Tensor [num_tiles, C, img_h, img_w]
  num_tiles is data-dependent (varies per image based on aspect ratio)

Step 2: Vision Encoding (GPU, at add_request time on PP stage 0)
────────────────────────────────────────────────────────────────
  Input: images [num_tiles, C, img_h, img_w]

  LLaVAModel.forward() lines 852-882:
    vision_model(images)        → [num_tiles, img_seq_len, h_vision]
      where img_seq_len = (img_h/patch_dim) * (img_w/patch_dim) + class_token
      e.g., CLIP 336/14: (24*24) + 1 = 577, or 576 if class token dropped

    Optional: drop class token  → [num_tiles, img_seq_len - 1, h_vision]
    Optional: pixel_shuffle     → [num_tiles, img_seq_len/4, h_vision*4]

    permute(1,0,2)              → [img_seq_len, num_tiles, h_vision]
    vision_projection(emb)      → [img_seq_len, num_tiles, h_language]

    Optional: _apply_tile_tagging()  [line 754]
      Prepend tile tag embeddings (<tile_1>, <tile_2>, ..., <tile_global_thumbnail>)
      Each tag is ~5 token IDs embedded via language_model.embedding()
      → [tile_tag_len + img_seq_len, num_tiles, h_language]

  Output: image_embeddings [final_img_seq_len, num_tiles, h_language]

Step 3: Sequence Expansion (at add_request time)
────────────────────────────────────────────────
  Input: prompt_token_ids with -200 placeholders, image_embeddings

  Expansion formula (from _preprocess_data, line 516):
    expanded_len = text_seq_len + sum(num_tiles_per_image * img_seq_len)
                   - num_image_placeholder_tokens

  get_num_image_embeddings()  [clip_vit_model.py:205] computes img_seq_len:
    base = (img_h / patch_dim) * (img_w / patch_dim)    # e.g., 576
    + class_token_len if kept                            # e.g., +1 = 577
    / 4 if pixel_shuffle                                 # e.g., 577/4 = 144
    + 5 or 6 if tile_tags (tokenizer-dependent)          # e.g., +5 = 149

  Example: prompt "Describe <image>" = 3 text tokens + 1 placeholder
           Image with 4 tiles, img_seq_len=149 (pixel_shuffle + tile_tags)
           expanded_len = 3 + (4 * 149) - 1 = 598 tokens in KV cache
```

### What needs to move into megatron/core/

For the REST API server to preprocess images, it needs access to image
preprocessing. Two options:

**Option A (recommended): Move `ImageTransform` into `megatron/core/inference/`**

Create `megatron/core/inference/multimodal/image_preprocessing.py` containing:
- `ImageTransform` class (from `examples/multimodal/image_processing.py`)
- `dynamic_preprocess()` function
- `find_closest_aspect_ratio()` function
- `_build_transform()` function
- The `pixel_statistics` dict

These are pure preprocessing utilities with no training dependencies — just
torchvision transforms. They belong in core.

**Option B: Import from examples/**

Add `examples/multimodal/` to `sys.path` from the server. Fragile and
not recommended for production.

### Vision model config needed at preprocessing time

The preprocessing pipeline requires these config values, which must be available
at server initialization (not per-request):

| Config | Where it comes from | Example |
|--------|-------------------|---------|
| `vision_model_type` | Model args | `"clip"`, `"siglip"`, `"internvit"` |
| `img_h` | Model args | `336`, `448` |
| `img_w` | Model args | `336`, `448` |
| `patch_dim` | Model args | `14` |
| `use_tiling` | Model args | `True`/`False` |
| `max_num_tiles` | Model args | `1`, `4`, `6`, `12` |
| `use_thumbnail` | Model args | `True`/`False` |
| `disable_vision_class_token` | Model args | `True`/`False` |
| `pixel_shuffle` | Model args | `True`/`False` |
| `use_tile_tags` | Model args | `True`/`False` |
| `tokenizer_prompt_format` | Model args | `"llama3p1"`, `"chatml"` |

These should be stored on the `VLMDynamicInferenceWrapper` (or a config object
attached to it) at initialization, and made accessible to the REST API endpoint
handler.

### Where preprocessing runs in each deployment mode

| Mode | Preprocessing location | Image data flow |
|------|----------------------|----------------|
| **Standalone script** | In the script, before `engine.add_request()` | File → PIL → `ImageTransform` → tensor → `add_request(images=...)` |
| **REST API (single node)** | In the endpoint handler | Base64/URL → PIL → `ImageTransform` → tensor → `client.add_request(images=...)` |
| **REST API + DP coordinator** | At the coordinator (before routing) | Base64/URL → PIL → `ImageTransform` → tensor → serialize → route to engine rank → `add_request(images=...)` |

In all cases, preprocessing happens BEFORE the request enters the dynamic engine.
The engine's `add_request()` receives already-preprocessed image tensors
(`[num_tiles, C, img_h, img_w]`) and the tile count.

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

## Encoder/Decoder Memory Sharing: How KV Cache Works

Understanding how the vision encoder output reaches the language decoder's KV
cache is central to the design. Here is the exact mechanism.

### Dynamic engine KV cache structure

The `DynamicInferenceContext` allocates a single contiguous GPU buffer for all
KV cache data:

```python
# dynamic_context.py:682
memory_buffer = torch.empty(
    (2,                          # 0=key, 1=value
     num_attention_layers,       # one slot per attention layer
     total_block_count,          # paged blocks
     block_size_tokens,          # tokens per block (e.g., 64)
     num_heads_per_partition,    # TP-sharded heads
     head_dim),                  # hidden_size / num_heads
    dtype=params_dtype,
)
```

Each request is assigned a set of block IDs. Each token position maps to a
specific `(block_id, position_within_block)` via:
- `token_to_block_idx[token_idx]` — which block
- `token_to_local_position_within_kv_block[token_idx]` — position within block

### How tokens write into the KV cache

During the forward pass, each attention layer calls:

```python
# dynamic_context.py:912
context.append_key_value_cache(layer_number, key, value)
```

This scatters the computed K and V tensors into the correct block positions:

```python
block_idx = token_to_block_idx[:padded_active_token_count]
local_pos = token_to_local_position_within_kv_block[:padded_active_token_count]
memory_buffer[0, layer, block_idx, local_pos] = key[:padded_active_token_count]
memory_buffer[1, layer, block_idx, local_pos] = value[:padded_active_token_count]
```

(A Triton kernel does this efficiently when available.)

### How this works for LLaVA with pre-expand strategy

The vision encoder and language decoder do NOT share memory directly. Instead,
the image embeddings flow through the language model's attention layers, which
write the resulting K/V into the shared KV cache. Here is the exact flow:

```
Prefill step for a request with images:

1. DynamicInferenceContext.add_request() is called with EXPANDED prompt tokens.
   The context allocates blocks for the full expanded length (text + image
   embedding positions). It sets up token_to_block_idx and
   token_to_local_position_within_kv_block for every position.

   Example: 3 text tokens + 596 image embedding positions = 599 tokens
   → allocates ceil(599/64) = 10 blocks

2. current_input_and_position_ids() returns the flat token_ids and position_ids
   for all active tokens (across all requests in the batch).

3. VLMDynamicInferenceWrapper._forward() receives these flat token_ids:
   a. Embeds ALL positions via language_model.embedding()
   b. Identifies image positions (the dummy token IDs at known offsets)
   c. Overwrites those positions with pre-computed image_embeddings
   d. Calls language_model(decoder_input=combined_embeddings)

4. Inside language_model, each transformer layer:
   a. Computes Q, K, V from the combined embeddings (both text AND image positions)
   b. Calls context.append_key_value_cache(layer, K, V)
   c. K and V for ALL positions (text and image alike) are scattered into
      their assigned blocks via token_to_block_idx

5. After prefill, the KV cache contains entries for every position:
   - Blocks 0-9 hold K/V for positions 0-598
   - Positions 0-595 hold K/V derived from image embeddings
   - Positions 596-598 hold K/V derived from text embeddings

6. On subsequent decode steps, the model generates one new token at a time.
   Attention reads from the existing KV cache blocks (which include the
   image-derived entries) via the block table. No image data is needed.
```

### Key insight: no "shared memory" between encoder and decoder

The vision encoder produces embeddings. These embeddings are spliced into the
token embedding stream. The language model's attention layers then process them
normally and write K/V into the paged KV cache. From the KV cache's perspective,
image-derived entries are indistinguishable from text-derived entries — they are
just K/V vectors at specific block positions.

The vision encoder never writes to the KV cache directly. The language model's
attention layers do all the writing. The "sharing" is simply: image embeddings
enter as input to the language model, and the language model's own attention
layers convert them to K/V cache entries.

### Parallelism implications

| Component | Where | Why |
|-----------|-------|-----|
| Vision encoder | PP stage 0, with existing TP | Only stage 0 has `add_encoder=True`; runs at `add_request()` time |
| Image embeddings (stored on request) | PP stage 0 GPU | Only stage 0 constructs `decoder_input` |
| Language model embedding layer | PP stage 0 (`pre_process=True`) | Embeds text tokens, then image positions are overwritten |
| Language model transformer layers | All PP stages via existing TP/PP | Process combined embeddings, write K/V to cache |
| KV cache (`memory_buffer`) | Each PP stage has its own | Each stage's layers write their own K/V; no cross-stage sharing needed |

The vision encoder is NOT replicated across PP stages. Only PP stage 0 runs it.
The combined embeddings flow through PP stages via the normal PP send/recv
mechanism (hidden states, not raw image data).

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
pass). This runs the same pipeline as `LLaVAModel.forward()` lines 852-882, but
pulled out into a standalone method on the wrapper:

```python
def compute_image_embeddings(self, images, num_image_tiles):
    """Run vision encoder + projection. Called at add_request() time on PP stage 0."""
    with torch.inference_mode():
        # Step 1: Vision encoder
        emb = self.model.vision_model(images)        # [num_tiles, img_seq_len, h_vision]

        # Step 2: Drop class token (if configured)
        if self._drop_vision_class_token:
            emb = emb[:, self.model.vision_model.class_token_len:, :]

        # Step 3: Pixel shuffle (if configured)
        if self._pixel_shuffle:
            emb = pixel_shuffle(emb)

        # Step 4: Permute and project
        emb = emb.permute(1, 0, 2).contiguous()      # [img_seq_len, num_tiles, h_vision]
        emb = self.model.vision_projection(emb)       # [img_seq_len, num_tiles, h_language]

        # Step 5: Tile tagging (if configured)
        if self.model._tile_tags is not None:
            emb = self.model._apply_tile_tagging(emb, num_image_tiles)
            # → [tile_tag_len + img_seq_len, num_tiles, h_language]

    return emb
```

Store the result as `request.image_embeddings`. This is a relatively cheap one-time
cost per request and avoids the complexity of batching vision encoding across
requests with different image counts.

**Important**: The tile tagging step (`_apply_tile_tagging`) calls
`language_model.embedding()` to embed the tile tag token IDs. This means the
language model's embedding layer must be available on the same rank that runs
vision encoding — which it is, because PP stage 0 has both `add_encoder=True`
and `pre_process=True`.

#### 1D. Computing the expanded sequence length

The caller must compute `num_image_embeddings_per_tile` using
`get_num_image_embeddings()` from `megatron/core/models/vision/clip_vit_model.py`
(line 205). This function accounts for all the optional transformations:

```python
num_img_embeddings_per_tile = get_num_image_embeddings(
    img_h, img_w, patch_dim, vision_model_type,
    disable_vision_class_token, class_token_len,
    pixel_shuffle, use_tile_tags, max_num_tiles, tokenizer_type
)

# For a request with images:
total_image_embeddings = sum(tiles_per_image * num_img_embeddings_per_tile
                            for tiles_per_image in num_image_tiles)
expanded_len = text_token_count + total_image_embeddings - num_placeholder_tokens
```

This value must match what `_preprocess_data()` would compute (line 516).

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

- `megatron/core/inference/multimodal/image_preprocessing.py` — `ImageTransform`, `dynamic_preprocess`, `_build_transform`, pixel stats (moved from `examples/multimodal/image_processing.py`)
- `megatron/core/inference/model_inference_wrappers/multimodal/vlm_dynamic_inference_wrapper.py` — `VLMDynamicInferenceWrapper` with `compute_image_embeddings()` and image-aware `_forward()`
- `examples/inference/vlm/vlm_dynamic_inference.py` — standalone entry point

## Files to Modify

- `megatron/core/inference/inference_request.py` — add image fields to `DynamicInferenceRequest`
- `megatron/core/inference/engines/dynamic_engine.py` — extend `add_request()` for images
- `megatron/core/inference/text_generation_server/dynamic_text_gen_server/endpoints/chat_completions.py` — multimodal messages
- `megatron/core/inference/data_parallel_inference_coordinator.py` — image data in coordinator protocol

## Files Unchanged (verification only)

- `megatron/core/inference/contexts/dynamic_context.py` — works with expanded tokens as-is
- `megatron/core/inference/text_generation_controllers/text_generation_controller.py` — `_dynamic_step_forward_logits` unchanged (wrapper handles image insertion internally)
- `megatron/core/models/multimodal/llava_model.py` — model unchanged, wrapper drives it differently
