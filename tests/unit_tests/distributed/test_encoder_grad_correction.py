# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from unittest import mock

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.param_and_grad_buffer import partition_buckets
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class EncoderTestModel(torch.nn.Module):
    """Model with separate encoder and backbone parameters for testing bucket isolation."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_encoder_layers: int,
        num_backbone_layers: int,
        bias: bool = False,
    ):
        super().__init__()
        self.encoder_layers = torch.nn.ModuleList(
            [torch.nn.Linear(input_dim, output_dim, bias) for _ in range(num_encoder_layers)]
        )
        self.backbone_layers = torch.nn.ModuleList(
            [torch.nn.Linear(input_dim, output_dim, bias) for _ in range(num_backbone_layers)]
        )

        # Mark encoder parameters.
        for param in self.encoder_layers.parameters():
            param.is_encoder_param = True


def get_encoder_model_and_buffers(
    input_dim: int = 100,
    output_dim: int = 100,
    num_encoder_layers: int = 2,
    num_backbone_layers: int = 3,
    bias: bool = False,
    bucket_size: int = None,
    use_distributed_optimizer: bool = False,
    overlap_grad_reduce: bool = False,
    average_in_collective: bool = False,
    correct_encoder_grad: bool = True,
):
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True,
        use_distributed_optimizer=use_distributed_optimizer,
        overlap_grad_reduce=overlap_grad_reduce,
        bucket_size=bucket_size,
        average_in_collective=average_in_collective,
        correct_encoder_grad_for_partial_participation=correct_encoder_grad,
    )
    model = EncoderTestModel(
        input_dim=input_dim,
        output_dim=output_dim,
        num_encoder_layers=num_encoder_layers,
        num_backbone_layers=num_backbone_layers,
        bias=bias,
    ).bfloat16()

    model = DistributedDataParallel(
        TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config=ddp_config, module=model
    )
    param_and_grad_buffer = model.buffers[0]
    bucket_groups = model.bucket_groups

    return model, param_and_grad_buffer, bucket_groups


def test_encoder_params_isolated_in_buckets():
    """Verify that encoder params are placed in separate buckets from backbone params."""
    Utils.initialize_model_parallel()

    model, param_and_grad_buffer, _ = get_encoder_model_and_buffers(
        num_encoder_layers=2,
        num_backbone_layers=3,
    )

    for bucket in param_and_grad_buffer.buckets:
        # Each bucket should be either all-encoder or all-backbone, never mixed.
        encoder_params = [p for p in bucket.params_list if getattr(p, 'is_encoder_param', False)]
        backbone_params = [p for p in bucket.params_list if not getattr(p, 'is_encoder_param', False)]
        assert len(encoder_params) == 0 or len(backbone_params) == 0, (
            f"Bucket {bucket.bucket_id} has mixed encoder ({len(encoder_params)}) "
            f"and backbone ({len(backbone_params)}) params"
        )

    # There should be at least one encoder bucket and one backbone bucket.
    encoder_buckets = [b for b in param_and_grad_buffer.buckets if b.is_encoder_bucket]
    backbone_buckets = [b for b in param_and_grad_buffer.buckets if not b.is_encoder_bucket]
    assert len(encoder_buckets) > 0, "Should have at least one encoder bucket"
    assert len(backbone_buckets) > 0, "Should have at least one backbone bucket"

    Utils.destroy_model_parallel()


@pytest.mark.parametrize("use_distributed_optimizer", [False, True])
@pytest.mark.parametrize("average_in_collective", [False, True])
def test_encoder_grad_correction_applied(
    use_distributed_optimizer: bool,
    average_in_collective: bool,
):
    """Verify that encoder bucket grad data is corrected after reduction."""
    Utils.initialize_model_parallel()

    model, param_and_grad_buffer, bucket_groups = get_encoder_model_and_buffers(
        use_distributed_optimizer=use_distributed_optimizer,
        average_in_collective=average_in_collective,
        correct_encoder_grad=True,
    )

    # Fill all grad data with 1.0.
    param_and_grad_buffer.grad_data.data.fill_(1.0)

    # Find encoder and backbone buckets.
    encoder_bucket_ids = set()
    backbone_bucket_ids = set()
    for bucket in param_and_grad_buffer.buckets:
        if bucket.is_encoder_bucket:
            encoder_bucket_ids.add(bucket.bucket_id)
        else:
            backbone_bucket_ids.add(bucket.bucket_id)

    dp_size = parallel_state.get_data_parallel_world_size()

    # Mock all-reduce and reduce-scatter so they're no-ops (single-process test).
    # Also mock the participation all-reduce to simulate partial participation:
    # simulate 2 out of dp_size ranks having encoder grads.
    # NOTE: In a single-process test, dp_size=1, so we need to mock
    # the collective_group_size to simulate a larger DP group.
    simulated_dp_size = 8
    simulated_participation = 2

    for bucket_group in bucket_groups:
        bucket_group.collective_group_size = simulated_dp_size

    # We need to intercept the participation all-reduce to inject our simulated count.
    original_all_reduce = torch.distributed.all_reduce

    def mock_all_reduce_side_effect(tensor, op=None, group=None, async_op=False):
        # If this is the scalar participation all-reduce (tensor is 1-element float32),
        # simulate partial participation.
        if tensor.numel() == 1 and tensor.dtype == torch.float32:
            tensor.fill_(simulated_participation)
            return None
        # Otherwise (grad reduction all-reduce), be a no-op.
        return None

    with mock.patch('torch.distributed.all_reduce', side_effect=mock_all_reduce_side_effect):
        with mock.patch(
            'megatron.core.distributed.param_and_grad_buffer.dist_reduce_scatter_func',
            return_value=None,
        ):
            model.finish_grad_sync()

    # Check: encoder buckets should have correction applied.
    # Expected correction: simulated_dp_size / simulated_participation = 8/2 = 4.0
    expected_correction = simulated_dp_size / simulated_participation

    for bucket in param_and_grad_buffer.buckets:
        if bucket.is_encoder_bucket:
            # Grad data in encoder buckets should be scaled.
            # Original was 1.0. With average_in_collective=False, gradient_scaling_factor
            # is 1/dp_size (but dp_size=1 in single process, so it's 1.0 here).
            # The mock all-reduce is a no-op on grad data, so the value stays at
            # gradient_scaling_factor * original. Then correction is applied.
            # gradient_scaling_factor = 1/1 = 1.0 (single process), so scaled = 1.0.
            # After correction: 1.0 * 4.0 = 4.0.
            expected_value = expected_correction
            if not average_in_collective:
                # gradient_scaling_factor = 1/dp_size = 1/1 = 1.0 (single process).
                expected_value = (1.0 / dp_size) * expected_correction
            assert torch.allclose(
                bucket.grad_data, torch.full_like(bucket.grad_data, expected_value)
            ), (
                f"Encoder bucket {bucket.bucket_id} grad_data should be {expected_value}, "
                f"got {bucket.grad_data[0].item()}"
            )
        else:
            # Backbone buckets should NOT have correction applied.
            expected_value = 1.0
            if not average_in_collective:
                expected_value = 1.0 / dp_size
            assert torch.allclose(
                bucket.grad_data, torch.full_like(bucket.grad_data, expected_value)
            ), (
                f"Backbone bucket {bucket.bucket_id} should be {expected_value}, "
                f"got {bucket.grad_data[0].item()}"
            )

    Utils.destroy_model_parallel()


def test_encoder_grad_correction_skipped_when_disabled():
    """Verify no correction when correct_encoder_grad_for_partial_participation=False."""
    Utils.initialize_model_parallel()

    model, param_and_grad_buffer, _ = get_encoder_model_and_buffers(correct_encoder_grad=False)

    param_and_grad_buffer.grad_data.data.fill_(1.0)

    with mock.patch('torch.distributed.all_reduce', return_value=None):
        model.finish_grad_sync()

    dp_size = parallel_state.get_data_parallel_world_size()
    # No correction should be applied. Value should just be gradient_scaling_factor * original.
    expected_value = 1.0 / dp_size
    for bucket in param_and_grad_buffer.buckets:
        assert torch.allclose(bucket.grad_data, torch.full_like(bucket.grad_data, expected_value)), (
            f"Bucket {bucket.bucket_id} should be {expected_value} (no correction), "
            f"got {bucket.grad_data[0].item()}"
        )

    Utils.destroy_model_parallel()


def test_encoder_grad_correction_all_ranks_participate():
    """When all ranks have encoder grads, no correction is needed (correction = 1.0)."""
    Utils.initialize_model_parallel()

    model, param_and_grad_buffer, bucket_groups = get_encoder_model_and_buffers()

    param_and_grad_buffer.grad_data.data.fill_(1.0)

    simulated_dp_size = 8

    for bucket_group in bucket_groups:
        bucket_group.collective_group_size = simulated_dp_size

    def mock_all_reduce_full_participation(tensor, op=None, group=None, async_op=False):
        if tensor.numel() == 1 and tensor.dtype == torch.float32:
            # All ranks participate.
            tensor.fill_(simulated_dp_size)
            return None
        return None

    with mock.patch(
        'torch.distributed.all_reduce', side_effect=mock_all_reduce_full_participation
    ):
        model.finish_grad_sync()

    dp_size = parallel_state.get_data_parallel_world_size()
    # correction = dp_size / dp_size = 1.0, so no change beyond gradient_scaling_factor.
    expected_value = 1.0 / dp_size
    for bucket in param_and_grad_buffer.buckets:
        assert torch.allclose(bucket.grad_data, torch.full_like(bucket.grad_data, expected_value)), (
            f"Bucket {bucket.bucket_id} should be {expected_value} (full participation), "
            f"got {bucket.grad_data[0].item()}"
        )

    Utils.destroy_model_parallel()


def test_encoder_grad_correction_zero_participation():
    """When no ranks have encoder grads (all zeros), no correction is applied."""
    Utils.initialize_model_parallel()

    model, param_and_grad_buffer, bucket_groups = get_encoder_model_and_buffers()

    # Set all grad data to zero (no rank has encoder grads).
    param_and_grad_buffer.grad_data.data.fill_(0.0)

    simulated_dp_size = 8

    for bucket_group in bucket_groups:
        bucket_group.collective_group_size = simulated_dp_size

    def mock_all_reduce_zero_participation(tensor, op=None, group=None, async_op=False):
        if tensor.numel() == 1 and tensor.dtype == torch.float32:
            tensor.fill_(0.0)
            return None
        return None

    with mock.patch(
        'torch.distributed.all_reduce', side_effect=mock_all_reduce_zero_participation
    ):
        model.finish_grad_sync()

    # All zeros, no correction. Values should remain 0.
    for bucket in param_and_grad_buffer.buckets:
        assert torch.allclose(bucket.grad_data, torch.zeros_like(bucket.grad_data)), (
            f"Bucket {bucket.bucket_id} should be all zeros, got {bucket.grad_data[0].item()}"
        )

    Utils.destroy_model_parallel()
