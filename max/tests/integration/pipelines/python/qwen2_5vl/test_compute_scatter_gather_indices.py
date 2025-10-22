# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


"""Test compute_scatter_gather_indices function."""

import copy

import numpy as np
from max.interfaces import ImageMetadata
from max.pipelines.architectures.qwen2_5vl.util import (
    compute_scatter_gather_indices,
)
from max.pipelines.core import TextAndVisionContext


def test_compute_scatter_gather_indices() -> None:
    IMG = 99
    # These pixel values are arbitrary
    img0 = np.array([[-1, -2], [-3, -4]])
    img1 = np.array([[-5, -6], [-7, -8]])
    ctx = TextAndVisionContext(
        max_length=50,
        tokens=np.array(
            [0, 1, 2, 3, IMG, IMG, IMG, IMG, 8, 9, IMG, IMG, IMG, IMG, IMG, 15]
        ),
        images=[
            ImageMetadata(
                start_idx=4,
                end_idx=8,
                pixel_values=img0,
            ),
            ImageMetadata(
                start_idx=10,
                end_idx=15,
                pixel_values=img1,
            ),
        ],
        vision_token_ids=[IMG],
        extra_model_args={
            "image_token_indices": np.array(
                [4, 5, 6, 7, 10, 11, 12, 13, 14], dtype=np.int32
            ),
        },
    )

    # fmt: off

    # Check that the image token indices are correct
    precomputed = ctx.extra_model_args["image_token_indices"]
    assert (precomputed == np.where(ctx.all_tokens == IMG)[0]).all()

    # Test normal case: start_idx = 0
    scatter_indices, gather_indices = compute_scatter_gather_indices([ctx])
    # 9 img tokens (img0 + img1)
    assert gather_indices.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    assert scatter_indices.tolist() == [4, 5, 6, 7, 10, 11, 12, 13, 14]

    # Test prefix cache hit case: start_idx = 8
    ctx.set_token_indices(start_idx=8)
    scatter_indices, gather_indices = compute_scatter_gather_indices([ctx])
    # 5 img tokens (img1)
    # 0-3 are skipped as img0 is not included
    assert gather_indices.tolist() == [4, 5, 6, 7, 8]
    assert scatter_indices.tolist() == [2, 3, 4, 5, 6]

    # Test multiple contexts case
    # ctx0 (start_idx=0), ctx1 (start_idx=8)
    ctx0 = copy.deepcopy(ctx)
    ctx1 = copy.deepcopy(ctx)
    ctx0.set_token_indices(start_idx=0)
    ctx1.set_token_indices(start_idx=8)
    scatter_indices, gather_indices = compute_scatter_gather_indices(
        [ctx0, ctx1]
    )
    # 9 (img0 + img1) + 5 (img1) = 14 img tokens
    # 9-12 are skipped as img0 of ctx1 is not included
    assert gather_indices.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17]
    assert scatter_indices.tolist() == [4, 5, 6, 7, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22]

    # Test empty case
    scatter_indices, gather_indices = compute_scatter_gather_indices([])
    assert scatter_indices.dtype == np.int32
    assert gather_indices.dtype == np.int64
    assert gather_indices.tolist() == []
    assert scatter_indices.tolist() == []
