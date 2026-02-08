# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from typing import Any, Dict, List, Optional, Tuple

import torch

from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)
from megatron.core.models.multimodal.llava_model import LLaVAModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import get_attr_wrapped_model


class VLMDynamicInferenceWrapper(GPTInferenceWrapper):
    """Inference wrapper for VLMs (e.g., LLaVA) with the dynamic inference engine.

    This wrapper handles image embedding insertion during the forward pass for
    dynamically-batched inference. Image embeddings are pre-computed at request-
    addition time and stored externally. During prefill, dummy token positions
    in the flat token buffer are replaced with the pre-computed image embeddings
    before passing them to the language model.

    Args:
        model: The LLaVA model (or compatible VLM).
        inference_wrapper_config: Has info like hidden size, vocab size, etc.
        inference_context: Manages KV cache and tracks sequence/token/batch offsets.
        pg_collection: Process groups for model communication.
    """

    def __init__(
        self,
        model: LLaVAModel,
        inference_wrapper_config: InferenceWrapperConfig,
        inference_context: Optional[BaseInferenceContext] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(model, inference_wrapper_config, inference_context, pg_collection)

        # Registry of image data for active requests.
        # Maps request_id -> (image_embeddings, image_token_positions)
        # image_embeddings: [img_seq_len, num_tiles, h_language]
        # image_token_positions: [(start, end), ...] in the expanded token sequence
        self._vlm_image_data: Dict[int, Tuple[torch.Tensor, List[Tuple[int, int]]]] = {}

        # Cache model attributes for vision encoding.
        unwrapped = get_attr_wrapped_model(model, "vision_model", return_model_obj=True)
        self._has_vision_model = hasattr(unwrapped, "vision_model")

    def register_image_data(
        self,
        request_id: int,
        image_embeddings: torch.Tensor,
        image_token_positions: List[Tuple[int, int]],
    ) -> None:
        """Register pre-computed image data for a request.

        Called by the engine at add_request() time.

        Args:
            request_id: The request's unique ID.
            image_embeddings: Pre-computed embeddings [img_seq_len, total_tiles, h_language].
            image_token_positions: (start, end) ranges in the expanded prompt sequence.
        """
        self._vlm_image_data[request_id] = (image_embeddings, image_token_positions)

    def unregister_image_data(self, request_id: int) -> None:
        """Remove image data for a completed/evicted request.

        Args:
            request_id: The request's unique ID.
        """
        self._vlm_image_data.pop(request_id, None)

    @property
    def has_pending_image_data(self) -> bool:
        """Check if any active requests have image data awaiting prefill."""
        return len(self._vlm_image_data) > 0

    def compute_image_embeddings(
        self,
        images: torch.Tensor,
        num_image_tiles: torch.Tensor,
    ) -> torch.Tensor:
        """Run the vision encoder + projection to produce image embeddings.

        This replicates LLaVAModel.forward() lines 852-882 but as a standalone
        method so it can be called at request-addition time.

        Args:
            images: Input image tiles [num_tiles, C, img_h, img_w].
            num_image_tiles: Number of tiles per image [num_images].

        Returns:
            Image embeddings [img_seq_len, total_tiles, h_language].
        """
        unwrapped = get_attr_wrapped_model(self.model, "vision_model", return_model_obj=True)

        with torch.inference_mode():
            # Vision encoder.
            image_embeddings = unwrapped.vision_model(images)  # [num_tiles, img_seq_len, h_vision]

            # Drop class token if configured.
            if getattr(unwrapped, "_drop_vision_class_token", False):
                class_token_len = unwrapped.vision_model.class_token_len
                image_embeddings = image_embeddings[:, class_token_len:, :]

            # Pixel shuffle if configured.
            if getattr(unwrapped, "_pixel_shuffle", False):
                from megatron.core.models.multimodal.llava_model import pixel_shuffle

                image_embeddings = pixel_shuffle(image_embeddings)

            # Permute and project.
            image_embeddings = image_embeddings.permute(1, 0, 2).contiguous()
            image_embeddings = unwrapped.vision_projection(image_embeddings)

            # Tile tagging if configured.
            if getattr(unwrapped, "_tile_tags", None) is not None:
                image_embeddings = unwrapped._apply_tile_tagging(
                    image_embeddings, num_image_tiles
                )

        return image_embeddings

    def _forward(self, inference_input: Dict[str, Any]):
        """Forward pass with image embedding insertion for VLM requests.

        For prefill steps where active requests have pre-computed image embeddings,
        this method:
        1. Embeds all tokens via the language model's embedding layer.
        2. Replaces dummy image positions with the pre-computed image embeddings.
        3. Calls the language model with decoder_input (bypassing its own embedding).

        For decode steps (or when no image data is pending), falls back to the
        standard GPT forward path.

        Args:
            inference_input: Dict with "tokens", "position_ids", "attention_mask".

        Returns:
            Logits tensor.
        """
        tokens = inference_input["tokens"]
        position_ids = inference_input["position_ids"]
        attention_mask = inference_input["attention_mask"]

        # Fast path: no pending image data, use standard GPT forward.
        if not self.has_pending_image_data:
            return super()._forward(inference_input)

        # Identify which requests in the current batch have image data.
        context = self.inference_context
        active_request_count = context.total_request_count - context.paused_request_count
        active_slice = slice(context.paused_request_count, context.total_request_count)
        active_request_ids = context.request_ids[active_slice].tolist()
        request_query_lengths = context.request_query_lengths[active_slice].tolist()

        # Check if any active request has pending image data.
        requests_with_images = {
            rid: self._vlm_image_data[rid]
            for rid in active_request_ids
            if rid in self._vlm_image_data
        }

        if not requests_with_images:
            # No image data for any active request in this step.
            return super()._forward(inference_input)

        # Get the unwrapped model for direct access to language_model.
        unwrapped = get_attr_wrapped_model(self.model, "language_model", return_model_obj=True)

        # Step 1: Embed all tokens via the language model's embedding layer.
        # Replace image placeholder token IDs (0 dummies) to avoid embedding issues.
        language_embeddings = unwrapped.language_model.embedding(
            input_ids=tokens, position_ids=position_ids
        )  # [seq_len, batch=1, h_language]

        # Step 2: Replace image positions with pre-computed embeddings.
        # The flat token buffer is laid out as:
        #   [request_0_tokens, request_1_tokens, ..., padding]
        # We need to map per-request image positions to flat buffer offsets.
        flat_offset = 0
        for i, (rid, query_len) in enumerate(zip(active_request_ids, request_query_lengths)):
            if rid in requests_with_images and query_len > 1:
                # This request is being prefilled and has image data.
                image_embeddings, image_token_positions = requests_with_images[rid]

                # image_token_positions are relative to the request's expanded prompt.
                # In the flat buffer, this request's tokens start at flat_offset.
                # For chunked prefill, we need to account for finished_chunk_token_count,
                # but for v1 we disable chunking for VLM requests, so positions are direct.
                img_emb_flat = image_embeddings.permute(1, 0, 2).reshape(
                    -1, image_embeddings.shape[-1]
                )  # [total_tiles * img_seq_len, h_language]

                for start, end in image_token_positions:
                    length = end - start
                    flat_start = flat_offset + start
                    flat_end = flat_offset + end
                    # Slice the relevant portion of the flattened image embeddings.
                    language_embeddings[flat_start:flat_end, 0, :] = img_emb_flat[:length]

            flat_offset += query_len

        # Step 3: Call language model with combined embeddings, bypassing its embedding layer.
        output = unwrapped.language_model(
            input_ids=None,
            position_ids=None,
            attention_mask=attention_mask,
            decoder_input=language_embeddings,
            inference_context=self.inference_context,
            runtime_gather_output=True,
        )

        # Note: we do NOT remove consumed image data here. It is kept around for
        # the lifetime of the request in case of eviction + re-prefill. The engine
        # calls unregister_image_data() when the request truly finishes.

        return output
