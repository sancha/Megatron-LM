# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compare VLM greedy decoding output between static and dynamic inference engines.

Loads a LLaVA model once, runs the same image+prompt through both:
  1. Static engine  (VLMInferenceWrapper + StaticInferenceEngine) — the existing path
  2. Dynamic engine (VLMDynamicInferenceWrapper + DynamicInferenceEngine) — the new path

Compares token-by-token greedy decoding output to verify correctness.

Usage:
    python examples/inference/vlm/compare_static_vs_dynamic.py \
        --load <checkpoint_path> \
        --input-image-path <image_file_or_dir> \
        --num-tokens-to-generate 64 \
        --use-tiling --max-num-tiles 4 \
        ...
"""

import os
import sys
from argparse import ArgumentParser
from functools import partial
from typing import List, Optional

import torch
from PIL import Image

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)

from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines import DynamicInferenceEngine, StaticInferenceEngine
from megatron.core.inference.inference_request import VLMInferenceRequest
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)
from megatron.core.inference.model_inference_wrappers.multimodal.vlm_dynamic_inference_wrapper import (
    VLMDynamicInferenceWrapper,
)
from megatron.core.inference.model_inference_wrappers.multimodal.vlm_inference_wrapper import (
    VLMInferenceWrapper,
)
from megatron.core.inference.multimodal.image_preprocessing import ImageTransform
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.inference.text_generation_controllers.vlm_text_generation_controller import (
    VLMTextGenerationController,
)
from megatron.core.models.multimodal.llava_model import IMAGE_TOKEN
from megatron.core.models.vision.clip_vit_model import get_num_image_embeddings
from megatron.core.tokenizers.text.utils.build_tokenizer import build_tokenizer
from megatron.core.transformer.module import MegatronModule

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)
from megatron.training import get_args, get_model as _get_model, get_tokenizer, initialize_megatron
from megatron.training.checkpointing import load_checkpoint

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "multimodal")
    )
)
from model import model_provider as vlm_model_provider
from multimodal_args import add_multimodal_extra_args


def add_comparison_args(parser: ArgumentParser) -> ArgumentParser:
    """Comparison test arguments."""
    add_multimodal_extra_args(parser)

    group = parser.add_argument_group(title="VLM comparison test")
    group.add_argument("--input-image-path", type=str, required=True)
    group.add_argument("--prompt", type=str, default="Describe this image in detail.")
    group.add_argument(
        "--prompt-format",
        type=str,
        default=None,
        help="Chat template. Use {prompt} and {image_token} as placeholders.",
    )
    group.add_argument("--num-tokens-to-generate", type=int, default=64)
    group.add_argument("--temperature", type=float, default=0.0)
    group.add_argument("--top-k", type=int, default=0)
    group.add_argument("--top-p", type=float, default=0.0)

    return parser


def get_model() -> MegatronModule:
    """Load LLaVA model."""
    args = get_args()
    model = _get_model(
        partial(vlm_model_provider, pre_process=True, post_process=True),
        wrap_with_ddp=False,
    )
    assert args.load is not None
    args.exit_on_missing_checkpoint = True
    load_checkpoint(ddp_model=model, optimizer=None, opt_param_scheduler=None, strict=True)
    assert len(model) == 1
    model = model[0]
    model.eval()
    return model


def preprocess_image(args, image_transform):
    """Load and preprocess a single image."""
    path = args.input_image_path
    if os.path.isdir(path):
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
        files = sorted(
            f for f in os.listdir(path) if os.path.splitext(f)[1].lower() in extensions
        )
        path = os.path.join(path, files[0])

    pil_image = Image.open(path).convert("RGB")
    tiles = image_transform(
        pil_image,
        args.img_h,
        args.img_w,
        use_tiling=args.use_tiling,
        max_num_tiles=args.max_num_tiles,
        use_thumbnail=args.use_thumbnail,
    )
    images = torch.stack(tiles).to(torch.cuda.current_device())
    num_tiles = torch.tensor([len(tiles)], dtype=torch.int, device=torch.cuda.current_device())
    return images, num_tiles, path


def build_prompt(args):
    """Build the text prompt with image token."""
    if args.prompt_format:
        return args.prompt_format.format(prompt=args.prompt, image_token=IMAGE_TOKEN)
    return f"{IMAGE_TOKEN}\n{args.prompt}"


# ──────────────────────────────────────────────
#  Static engine path (existing VLMInferenceWrapper)
# ──────────────────────────────────────────────
def run_static_inference(
    model: MegatronModule,
    prompt: str,
    images: torch.Tensor,
    num_tiles: torch.Tensor,
    num_img_embeddings_per_tile: int,
    sampling_params: SamplingParams,
) -> List[int]:
    """Run inference through the existing static VLM engine. Returns generated token IDs."""
    args = get_args()
    tokenizer = get_tokenizer() if args.legacy_tokenizer else build_tokenizer(args)

    wrapper_config = InferenceWrapperConfig(
        hidden_size=args.hidden_size,
        inference_batch_times_seqlen_threshold=args.inference_batch_times_seqlen_threshold,
        fp32_residual_connection=args.fp32_residual_connection,
        params_dtype=args.params_dtype,
        padded_vocab_size=args.padded_vocab_size,
    )
    wrapped_model = VLMInferenceWrapper(model, wrapper_config)
    controller = VLMTextGenerationController(
        inference_wrapped_model=wrapped_model, tokenizer=tokenizer
    )
    engine = StaticInferenceEngine(controller, max_batch_size=1, random_seed=args.seed, legacy=True)

    prompt_tokens = controller.tokenize_prompt(prompt)
    request = VLMInferenceRequest(
        request_id=engine.get_new_request_id(),
        prompt=prompt,
        prompt_tokens=prompt_tokens,
        sampling_params=sampling_params,
        num_img_embeddings_per_tile=num_img_embeddings_per_tile,
        imgs=images,
        num_tiles=num_tiles,
        decoder_seq_length=args.decoder_seq_length,
    )
    results = engine.generate(inference_requests=[request])
    result = results[0]
    return result.generated_tokens, result.generated_text


# ──────────────────────────────────────────────
#  Dynamic engine path (new VLMDynamicInferenceWrapper)
# ──────────────────────────────────────────────
def run_dynamic_inference(
    model: MegatronModule,
    prompt: str,
    images: torch.Tensor,
    num_tiles: torch.Tensor,
    num_img_embeddings_per_tile: int,
    sampling_params: SamplingParams,
) -> List[int]:
    """Run inference through the new dynamic VLM engine. Returns generated token IDs."""
    args = get_args()
    tokenizer = get_tokenizer() if args.legacy_tokenizer else build_tokenizer(args)

    # Pre-compute image embeddings.
    temp_wrapper = VLMDynamicInferenceWrapper(model, args, None)
    image_embeddings = temp_wrapper.compute_image_embeddings(images, num_tiles)

    # Compute expanded prompt length for max_sequence_length.
    prompt_tokens = tokenizer.tokenize(prompt) if hasattr(tokenizer, "tokenize") else tokenizer.encode(prompt)
    num_placeholders = sum(1 for t in prompt_tokens if t == -200)
    total_tiles = num_tiles.sum().item()
    expanded_len = len(prompt_tokens) + total_tiles * num_img_embeddings_per_tile - num_placeholders
    max_seq_len = expanded_len + sampling_params.num_tokens_to_generate

    # Build context.
    context = DynamicInferenceContext(
        params_dtype=args.params_dtype,
        num_layers=args.num_layers // args.pipeline_model_parallel_size,
        kv_channels=args.kv_channels,
        num_attention_heads=(
            args.num_query_groups if args.group_query_attention else args.num_attention_heads
        ),
        max_sequence_length=max_seq_len,
        num_cuda_graphs=None,
        block_size_tokens=args.inference_dynamic_batching_block_size,
        buffer_size_gb=args.inference_dynamic_batching_buffer_size_gb,
        paused_buffer_size_gb=getattr(args, "inference_dynamic_batching_paused_buffer_size_gb", 0),
        max_requests=args.inference_dynamic_batching_max_requests,
        max_tokens=args.inference_dynamic_batching_max_tokens,
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        materialize_only_last_token_logits=True,
        use_cuda_graphs_for_non_decode_steps=False,
    )

    # Build wrapper and controller.
    wrapped_model = VLMDynamicInferenceWrapper(model, args, context)
    from megatron.core import parallel_state

    wrapped_model.model_is_pipeline_parallel = not (
        parallel_state.is_pipeline_first_stage() and parallel_state.is_pipeline_last_stage()
    )
    controller = TextGenerationController(wrapped_model, tokenizer)

    # Build engine.
    engine = DynamicInferenceEngine(
        controller,
        context,
        enable_cuda_graph=False,
        random_seed=args.seed,
        enable_chunked_prefill=False,
    )

    # Add request with pre-computed image embeddings.
    prompt_tensor = torch.tensor(
        prompt_tokens, dtype=torch.int64, device=torch.cuda.current_device()
    )
    engine.add_request(
        request_id=0,
        prompt=prompt_tensor,
        sampling_params=sampling_params,
        image_embeddings=image_embeddings,
    )

    # Run inference loop.
    generated_tokens = None
    generated_text = None
    while engine.has_unfinished_requests():
        result = engine.step_modern()
        for record in result["finished_request_records"]:
            merged = record.merge()
            generated_tokens = merged.generated_tokens
            generated_text = merged.generated_text

    return generated_tokens, generated_text


# ──────────────────────────────────────────────
#  Main: run both and compare
# ──────────────────────────────────────────────
@torch.inference_mode()
def main():
    initialize_megatron(
        extra_args_provider=add_comparison_args,
        args_defaults={"no_load_rng": True, "no_load_optim": True},
    )

    args = get_args()
    rank = torch.distributed.get_rank()

    # Greedy decoding.
    sampling_params = SamplingParams(
        temperature=0.0,
        top_k=0,
        top_p=0.0,
        num_tokens_to_generate=args.num_tokens_to_generate,
    )

    # Load model once.
    model = get_model()

    # Preprocess image.
    image_transform = ImageTransform(args.img_h, args.vision_model_type)
    images, num_tiles, image_path = preprocess_image(args, image_transform)

    num_img_embeddings_per_tile = get_num_image_embeddings(
        args.img_h,
        args.img_w,
        args.patch_dim,
        args.vision_model_type,
        args.disable_vision_class_token,
        1,
        args.pixel_shuffle,
        args.use_tile_tags,
        args.max_num_tiles,
        args.tokenizer_prompt_format,
    )

    prompt = build_prompt(args)

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"VLM Inference Comparison: Static vs Dynamic")
        print(f"{'='*60}")
        print(f"Image:  {image_path}")
        print(f"Tiles:  {num_tiles.item()}")
        print(f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        print(f"Img embeddings/tile: {num_img_embeddings_per_tile}")
        print(f"Max tokens to generate: {args.num_tokens_to_generate}")
        print(f"{'='*60}")

    # ── Run static engine ──
    if rank == 0:
        print("\n[1/2] Running STATIC inference engine...")
    static_tokens, static_text = run_static_inference(
        model, prompt, images, num_tiles, num_img_embeddings_per_tile, sampling_params
    )
    if rank == 0:
        print(f"  Generated {len(static_tokens)} tokens")

    # ── Run dynamic engine ──
    if rank == 0:
        print("\n[2/2] Running DYNAMIC inference engine...")
    dynamic_tokens, dynamic_text = run_dynamic_inference(
        model, prompt, images, num_tiles, num_img_embeddings_per_tile, sampling_params
    )
    if rank == 0:
        print(f"  Generated {len(dynamic_tokens)} tokens")

    # ── Compare ──
    if rank == 0:
        print(f"\n{'='*60}")
        print("COMPARISON RESULTS")
        print(f"{'='*60}")

        print(f"\nStatic  output ({len(static_tokens)} tokens): {static_text}")
        print(f"\nDynamic output ({len(dynamic_tokens)} tokens): {dynamic_text}")

        # Token-by-token comparison.
        min_len = min(len(static_tokens), len(dynamic_tokens))
        max_len = max(len(static_tokens), len(dynamic_tokens))
        mismatches = []
        for i in range(min_len):
            if static_tokens[i] != dynamic_tokens[i]:
                mismatches.append((i, static_tokens[i], dynamic_tokens[i]))

        if len(static_tokens) != len(dynamic_tokens):
            print(f"\n  LENGTH MISMATCH: static={len(static_tokens)}, dynamic={len(dynamic_tokens)}")

        if mismatches:
            print(f"\n  TOKEN MISMATCHES: {len(mismatches)} out of {min_len} tokens differ")
            for pos, s_tok, d_tok in mismatches[:10]:
                print(f"    Position {pos}: static={s_tok}, dynamic={d_tok}")
            if len(mismatches) > 10:
                print(f"    ... and {len(mismatches) - 10} more")
        else:
            print(f"\n  All {min_len} tokens MATCH")

        if not mismatches and len(static_tokens) == len(dynamic_tokens):
            print(f"\n  PASS: Static and dynamic engines produce identical greedy output.")
        else:
            print(f"\n  FAIL: Outputs differ. Debug needed.")

        print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
