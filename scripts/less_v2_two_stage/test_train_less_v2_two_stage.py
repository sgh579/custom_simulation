#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn.functional as F

from train_less_v2_two_stage import (
    ParticleTCNN2D,
    ParticleTCNN3D,
    build_patch_placement,
    scan_area_label_xy,
    stitch_patches,
)


class DecoderGeometryTests(unittest.TestCase):
    def test_low_resolution_stitch_matches_fold(self) -> None:
        generator = torch.Generator().manual_seed(9400)
        patches = torch.randn(2, 3, 400, 25, generator=generator)
        indices, valid = build_patch_placement(
            particle_height=20,
            particle_width=20,
            output_height=20,
            output_width=20,
            patch_size=5,
        )
        actual = stitch_patches(
            patches,
            placement_indices=indices,
            placement_valid=valid,
            height=20,
            width=20,
        )
        expected = torch.stack(
            [
                F.fold(
                    patches[:, channel].transpose(1, 2),
                    output_size=(20, 20),
                    kernel_size=5,
                    padding=2,
                )[:, 0]
                for channel in range(patches.shape[1])
            ],
            dim=1,
        )
        torch.testing.assert_close(actual, expected)

    def test_high_resolution_2d_decoder_shape_and_gradient(self) -> None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = ParticleTCNN2D(
            representation_dim=8,
            channels=8,
            patch_size=32,
            output_size=128,
        ).to(device)
        representations = torch.randn(1, 400, 8, device=device)
        output = model(representations)
        self.assertEqual(tuple(output.shape), (1, 128, 128))
        self.assertTrue(bool(torch.isfinite(output).all()))
        output.square().mean().backward()
        self.assertIsNotNone(model.project.weight.grad)
        self.assertTrue(bool(torch.isfinite(model.project.weight.grad).all()))

    def test_high_resolution_3d_decoder_shape(self) -> None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = ParticleTCNN3D(
            representation_dim=8,
            channels=8,
            volume_depth=16,
            patch_size=32,
            output_size=128,
        ).to(device)
        representations = torch.randn(1, 400, 8, device=device)
        with torch.no_grad():
            output = model(representations)
        self.assertEqual(tuple(output.shape), (1, 16, 128, 128))
        self.assertTrue(bool(torch.isfinite(output).all()))

    def test_scan_area_grid_matches_grid_unet_convention(self) -> None:
        xs = np.linspace(-0.075, 0.075, 20, dtype=np.float32)
        ys = np.linspace(-0.060, 0.060, 20, dtype=np.float32)
        native = np.stack(np.meshgrid(xs, ys), axis=-1)
        target = scan_area_label_xy(native, 128)
        self.assertEqual(target.shape, (128, 128, 2))
        np.testing.assert_array_equal(
            target[0, :, 0],
            np.linspace(float(xs[0]), float(xs[-1]), 128, dtype=np.float32),
        )
        np.testing.assert_array_equal(
            target[:, 0, 1],
            np.linspace(float(ys[0]), float(ys[-1]), 128, dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
