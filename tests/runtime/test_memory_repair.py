from __future__ import annotations

import itertools
import unittest
from collections.abc import Callable
from unittest import mock

import torch
import torch.nn.functional as functional

from apps.backend.runtime.attention import attention_function_single_head_spatial
from apps.backend.runtime.common.vae_codex3d import (
    _CODEX3D_TILED_ATTENTION_QUERY_TOKENS,
    AutoencoderCodex3D,
    Codex3DAttentionBlock,
)
from apps.backend.runtime.vision.upscalers import tiled_scale as tiled_scale_module


def _legacy_single_head_spatial_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = q.shape
    query = q.reshape(batch, channels, height * width).permute(0, 2, 1)
    key = k.reshape(batch, channels, height * width)
    value = v.reshape(batch, channels, height * width)
    scores = torch.bmm(query, key) * (channels ** -0.5)
    weights = torch.nn.functional.softmax(scores, dim=-1).permute(0, 2, 1)
    return torch.bmm(value, weights).reshape(batch, channels, height, width)


def _legacy_tiled_scale(
    samples: torch.Tensor,
    function: Callable[[torch.Tensor], torch.Tensor],
    *,
    tile: tuple[int, int],
    overlap: int,
    upscale_amount: float,
    out_channels: int,
    output_device: str | torch.device,
) -> torch.Tensor:
    dims = len(tile)
    scaled_spatial = [round(size * upscale_amount) for size in samples.shape[2:]]
    output = torch.empty([samples.shape[0], out_channels] + scaled_spatial, device=output_device)

    for batch_index in range(int(samples.shape[0])):
        sample = samples[batch_index : batch_index + 1]
        out = torch.zeros([1, out_channels] + scaled_spatial, device=output_device)
        out_div = torch.zeros_like(out)
        ranges = [range(0, int(shape), int(tile_size) - int(overlap)) for shape, tile_size in zip(sample.shape[2:], tile, strict=True)]

        for tile_start in itertools.product(*ranges):
            tile_input = sample
            upscaled = []
            for dimension in range(dims):
                position = max(0, min(int(sample.shape[dimension + 2]) - int(overlap), int(tile_start[dimension])))
                length = min(int(tile[dimension]), int(sample.shape[dimension + 2]) - position)
                tile_input = tile_input.narrow(dimension + 2, position, length)
                upscaled.append(round(position * upscale_amount))

            prediction = function(tile_input).to(output_device)
            mask = torch.ones_like(prediction)
            feather = round(int(overlap) * float(upscale_amount))
            for offset in range(int(feather)):
                for dimension in range(2, dims + 2):
                    leading = mask.narrow(dimension, offset, 1)
                    leading *= ((1.0 / feather) * (offset + 1))
                    trailing = mask.narrow(dimension, int(mask.shape[dimension]) - 1 - offset, 1)
                    trailing *= ((1.0 / feather) * (offset + 1))

            out_region = out
            weight_region = out_div
            for dimension in range(dims):
                out_region = out_region.narrow(dimension + 2, int(upscaled[dimension]), int(mask.shape[dimension + 2]))
                weight_region = weight_region.narrow(dimension + 2, int(upscaled[dimension]), int(mask.shape[dimension + 2]))
            out_region += prediction * mask
            weight_region += mask

        output[batch_index : batch_index + 1] = out / out_div

    return output


class RuntimeMemoryRepairTests(unittest.TestCase):
    def test_codex3d_tiling_propagates_exact_query_tiling_without_changing_attention_result(self) -> None:
        torch.manual_seed(7)
        q = torch.randn((1, 4, 7, 9), dtype=torch.float32)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        expected = _legacy_single_head_spatial_attention(q, k, v)
        actual = attention_function_single_head_spatial(q, k, v, query_chunk_size=5)
        torch.testing.assert_close(actual, expected)

        block = Codex3DAttentionBlock(dim=4).eval()
        block_input = torch.randn((1, 4, 2, 7, 9), dtype=torch.float32)
        with torch.inference_mode():
            full_attention = block(block_input)
            block.enable_query_tiling(query_chunk_size=5)
            query_tiled_attention = block(block_input)
        torch.testing.assert_close(query_tiled_attention, full_attention)

        vae = AutoencoderCodex3D(
            base_dim=4,
            z_dim=2,
            dim_mult=(1,),
            num_res_blocks=1,
            attn_scales=(),
            temperal_downsample=(),
        )
        attention_blocks = tuple(module for module in vae.modules() if isinstance(module, Codex3DAttentionBlock))
        self.assertTrue(attention_blocks)
        self.assertTrue(all(block._query_chunk_size is None for block in attention_blocks))

        vae.enable_tiling()

        self.assertTrue(vae.use_tiling)
        self.assertEqual(
            {block._query_chunk_size for block in attention_blocks},
            {_CODEX3D_TILED_ATTENTION_QUERY_TOKENS},
        )

    def test_tiled_scale_reuses_final_output_and_single_channel_blend_weights(self) -> None:
        torch.manual_seed(11)
        samples = torch.rand((1, 3, 7, 9), dtype=torch.float32)
        tile = (4, 5)
        overlap = 1
        upscale_amount = 2.0

        def upscale(tile_input: torch.Tensor) -> torch.Tensor:
            return functional.interpolate(tile_input, scale_factor=upscale_amount, mode="nearest")

        expected = _legacy_tiled_scale(
            samples,
            upscale,
            tile=tile,
            overlap=overlap,
            upscale_amount=upscale_amount,
            out_channels=3,
            output_device="cpu",
        )

        original_zeros = torch.zeros
        original_zeros_like = torch.zeros_like
        allocations: list[tuple[str, tuple[int, ...]]] = []

        def record_zeros(*args, **kwargs):
            shape = tuple(int(size) for size in args[0])
            allocations.append(("zeros", shape))
            return original_zeros(*args, **kwargs)

        def record_zeros_like(tensor: torch.Tensor, *args, **kwargs):
            allocations.append(("zeros_like", tuple(int(size) for size in tensor.shape)))
            return original_zeros_like(tensor, *args, **kwargs)

        with (
            mock.patch.object(tiled_scale_module.torch, "zeros", record_zeros),
            mock.patch.object(tiled_scale_module.torch, "zeros_like", record_zeros_like),
        ):
            actual = tiled_scale_module.tiled_scale_multidim(
                samples,
                upscale,
                tile=tile,
                overlap=overlap,
                upscale_amount=upscale_amount,
                out_channels=3,
                output_device="cpu",
            )

        torch.testing.assert_close(actual, expected)
        self.assertNotIn(("zeros_like", (1, 3, 14, 18)), allocations)
        self.assertEqual(allocations.count(("zeros", (1, 3, 14, 18))), 1)
        self.assertEqual(allocations.count(("zeros", (1, 1, 14, 18))), 1)


if __name__ == "__main__":
    unittest.main()
