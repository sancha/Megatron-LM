# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Dynamic inference for Vision Language Models (LLaVA).

This script provides the same high-throughput dynamic batching inference as
gpt_dynamic_inference.py, but for multimodal models (LLaVA). It pre-computes
vision embeddings at request-addition time and uses the VLMDynamicInferenceWrapper
to splice them into the token stream during prefill.

Usage:
    python examples/inference/vlm/vlm_dynamic_inference.py \
        --load <checkpoint_path> \
        --input-image-path <image_dir> \
        --num-tokens-to-generate 128 \
        ...
"""

import json
import os
import sys
from argparse import ArgumentParser
from functools import partial
from typing import Dict, List, Optional

import torch
from PIL import Image

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)

from examples.inference.gpt.utils import (
    add_common_inference_args,
    get_curr_time,
)
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.model_inference_wrappers.multimodal.vlm_dynamic_inference_wrapper import (
    VLMDynamicInferenceWrapper,
)
from megatron.core.inference.multimodal.image_preprocessing import ImageTransform
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
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

# Model provider is expected from the examples/multimodal directory.
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "multimodal"))
)
from model import model_provider as vlm_model_provider
from multimodal_args import add_multimodal_extra_args


def add_vlm_dynamic_inference_args(parser: ArgumentParser) -> ArgumentParser:
    """VLM dynamic inference arguments."""

    add_common_inference_args(parser)
    add_multimodal_extra_args(parser)

    group = parser.add_argument_group(title='VLM Dynamic inference')
    group.add_argument(
        "--input-image-path", type=str, required=True,
        help="Path to input image or directory of images.",
    )
    group.add_argument(
        "--prompts", type=str, nargs="+",
        default=["Describe this image in detail."],
        help="Text prompts to use with each image.",
    )
    group.add_argument(
        "--prompt-format", type=str, default=None,
        help="Chat template format string. Use {prompt} and {image_token} as placeholders.",
    )

    return parser


def get_model() -> MegatronModule:
    """Initialize LLaVA model and load checkpoint."""
    args = get_args()

    model = _get_model(
        partial(vlm_model_provider, pre_process=True, post_process=True),
        wrap_with_ddp=False,
    )

    assert args.load is not None
    args.exit_on_missing_checkpoint = True
    load_checkpoint(
        ddp_model=model,
        optimizer=None,
        opt_param_scheduler=None,
        strict=True,
    )

    assert len(model) == 1
    model = model[0]
    model.eval()

    return model


def build_image_requests(
    args,
    tokenizer,
    sampling_params: SamplingParams,
    image_transform: ImageTransform,
) -> List[Dict]:
    """Build inference requests from images and prompts.

    Returns a list of dicts with keys: prompt_text, prompt_tokens, images, num_image_tiles.
    """
    # Collect image paths.
    image_path = args.input_image_path
    if os.path.isdir(image_path):
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
        image_paths = sorted([
            os.path.join(image_path, f)
            for f in os.listdir(image_path)
            if os.path.splitext(f)[1].lower() in extensions
        ])
    else:
        image_paths = [image_path]

    prompts = args.prompts

    requests = []
    for img_path in image_paths:
        for prompt_text in prompts:
            # Preprocess image.
            pil_image = Image.open(img_path).convert("RGB")
            tiles = image_transform(
                pil_image,
                args.img_h,
                args.img_w,
                use_tiling=args.use_tiling,
                max_num_tiles=args.max_num_tiles,
                use_thumbnail=args.use_thumbnail,
            )
            images = torch.stack(tiles).to(torch.cuda.current_device())
            num_image_tiles = torch.tensor(
                [len(tiles)], dtype=torch.int, device=torch.cuda.current_device()
            )

            # Build the prompt with image token.
            if args.prompt_format:
                full_prompt = args.prompt_format.format(
                    prompt=prompt_text, image_token=IMAGE_TOKEN
                )
            else:
                full_prompt = f"{IMAGE_TOKEN}\n{prompt_text}"

            # Tokenize.
            try:
                prompt_tokens = tokenizer.tokenize(full_prompt)
            except AttributeError:
                prompt_tokens = tokenizer.encode(full_prompt)

            requests.append({
                "prompt_text": full_prompt,
                "prompt_tokens": prompt_tokens,
                "images": images,
                "num_image_tiles": num_image_tiles,
                "image_path": img_path,
            })

    return requests


def get_inference_context(
    expanded_lengths: List[int],
    sampling_params: SamplingParams,
) -> DynamicInferenceContext:
    """Build the dynamic inference context with VLM-aware max sequence length."""
    args = get_args()

    max_gen_length = sampling_params.num_tokens_to_generate
    max_context_length = max(expanded_lengths)
    max_sequence_length = max_context_length + max_gen_length

    context = DynamicInferenceContext(
        params_dtype=args.params_dtype,
        num_layers=args.num_layers // args.pipeline_model_parallel_size,
        kv_channels=args.kv_channels,
        num_attention_heads=(
            args.num_query_groups if args.group_query_attention else args.num_attention_heads
        ),
        max_sequence_length=max_sequence_length,
        num_cuda_graphs=(
            args.inference_dynamic_batching_num_cuda_graphs
            if args.cuda_graph_impl == "local"
            else None
        ),
        block_size_tokens=args.inference_dynamic_batching_block_size,
        buffer_size_gb=args.inference_dynamic_batching_buffer_size_gb,
        paused_buffer_size_gb=getattr(args, "inference_dynamic_batching_paused_buffer_size_gb", 0),
        max_requests=args.inference_dynamic_batching_max_requests,
        max_tokens=args.inference_dynamic_batching_max_tokens,
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        materialize_only_last_token_logits=True,
        # Disable non-decode CUDA graphs for VLM (image embedding insertion has variable shapes).
        use_cuda_graphs_for_non_decode_steps=False,
    )

    return context


def get_inference_controller(
    model: MegatronModule, context: DynamicInferenceContext
) -> TextGenerationController:
    """Build text generation controller with VLM wrapper."""
    args = get_args()
    if args.legacy_tokenizer:
        tokenizer = get_tokenizer()
    else:
        tokenizer = build_tokenizer(args)

    # Wrap model in VLM dynamic inference wrapper.
    model = VLMDynamicInferenceWrapper(model, args, context)

    from megatron.core import parallel_state
    model.model_is_pipeline_parallel = not (
        parallel_state.is_pipeline_first_stage() and parallel_state.is_pipeline_last_stage()
    )

    controller = TextGenerationController(model, tokenizer)
    return controller


@torch.inference_mode()
def main():
    # Initialize Megatron.
    initialize_megatron(
        extra_args_provider=add_vlm_dynamic_inference_args,
        args_defaults={'no_load_rng': True, 'no_load_optim': True},
    )

    args = get_args()
    if args.legacy_tokenizer:
        tokenizer = get_tokenizer()
    else:
        tokenizer = build_tokenizer(args)

    torch.cuda.reset_peak_memory_stats()

    # Sampling params.
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        num_tokens_to_generate=args.num_tokens_to_generate,
        termination_id=tokenizer.eod,
    )

    # Build model.
    model = get_model()

    # Image preprocessing.
    image_transform = ImageTransform(args.img_h, args.vision_model_type)

    # Compute how many embeddings each tile produces.
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

    # Build requests with image preprocessing.
    requests = build_image_requests(args, tokenizer, sampling_params, image_transform)

    # Create VLM wrapper to pre-compute image embeddings.
    # We need a temporary wrapper just for compute_image_embeddings.
    temp_wrapper = VLMDynamicInferenceWrapper(model, args, None)

    # Pre-compute image embeddings and compute expanded prompt lengths.
    expanded_lengths = []
    for req in requests:
        # Pre-compute vision embeddings.
        image_embeddings = temp_wrapper.compute_image_embeddings(
            req["images"], req["num_image_tiles"]
        )
        req["image_embeddings"] = image_embeddings

        # Compute expanded length.
        prompt_tokens = req["prompt_tokens"]
        num_placeholders = sum(1 for t in prompt_tokens if t == -200)
        total_tiles = req["num_image_tiles"].sum().item()
        expanded_len = (
            len(prompt_tokens)
            + total_tiles * num_img_embeddings_per_tile
            - num_placeholders
        )
        expanded_lengths.append(expanded_len)

    # Build context and controller.
    context = get_inference_context(expanded_lengths, sampling_params)
    controller = get_inference_controller(model, context)

    # Build engine.
    # Disable chunked prefill for VLM in v1 to avoid splitting image embedding regions.
    engine = DynamicInferenceEngine(
        controller,
        context,
        enable_cuda_graph=args.cuda_graph_impl == "local",
        random_seed=args.seed,
        enable_chunked_prefill=False,
    )

    # Add requests and generate.
    t_start = get_curr_time()

    for i, req in enumerate(requests):
        prompt_tokens = torch.tensor(
            req["prompt_tokens"], dtype=torch.int64, device=torch.cuda.current_device()
        )
        engine.add_request(
            request_id=i,
            prompt=prompt_tokens,
            sampling_params=sampling_params,
            image_embeddings=req["image_embeddings"],
        )

    # Run inference loop.
    total_output_tokens = 0
    while engine.has_unfinished_requests():
        result = engine.step_modern()
        finished_records = result["finished_request_records"]
        for record in finished_records:
            merged = record.merge()
            req = requests[merged.request_id]
            req["output_text"] = merged.generated_text
            req["output_tokens"] = merged.generated_tokens
            total_output_tokens += len(merged.generated_tokens)

    torch.cuda.synchronize()
    total_time = get_curr_time() - t_start

    # Print results.
    if torch.distributed.get_rank() == 0:
        for i, req in enumerate(requests):
            print(f"\n--- Request {i} ---")
            print(f"Image: {req['image_path']}")
            print(f"Prompt: {req['prompt_text']}")
            print(f"Output: {req.get('output_text', '<no output>')}")

        throughput = total_output_tokens / total_time if total_time > 0 else 0
        print(f"\n--- Summary ---")
        print(f"Total requests: {len(requests)}")
        print(f"Total output tokens: {total_output_tokens}")
        print(f"Total time: {total_time:.3f}s")
        print(f"Throughput: {throughput:.1f} tok/s")

        # Save results to JSON if output path is specified.
        if args.output_path:
            results = {}
            for i, req in enumerate(requests):
                results[i] = {
                    "image_path": req["image_path"],
                    "prompt": req["prompt_text"],
                    "generated_text": req.get("output_text"),
                    "generated_tokens": req.get("output_tokens"),
                }
            with open(args.output_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    main()
