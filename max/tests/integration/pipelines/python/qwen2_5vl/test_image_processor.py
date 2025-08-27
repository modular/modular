# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


"""Simple test to compare Qwen2_5VLImageProcessor with transformers library using real images."""

from __future__ import annotations

import io
import urllib.request

import numpy as np
from max.pipelines.architectures.qwen2_5vl.nn.qwen_vl_utils import (
    fetch_image,
)

# Import our custom processor
from max.pipelines.architectures.qwen2_5vl.tokenizer import (
    Qwen2_5VLImageProcessor,
)
from PIL import Image
from transformers import Qwen2VLImageProcessor


def download_image(url: str) -> Image.Image:
    """Download an image from URL and apply the same preprocessing as transformers library."""
    print(f"Downloading image from: {url}")
    # First download the raw image
    with urllib.request.urlopen(url) as response:
        image_data = response.read()
    raw_image = Image.open(io.BytesIO(image_data))

    print(f"Raw image size: {raw_image.size}")

    # Apply the same preprocessing as transformers library using fetch_image

    processed_image = fetch_image({"image": raw_image})

    print(f"Processed image size: {processed_image.size}")
    return processed_image


def test_image_processor():
    """Compare our custom processor with transformers library."""
    # Download the demo image
    image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    test_image = download_image(image_url)

    print(f"Original image size: {test_image.size}")
    print(f"Original image mode: {test_image.mode}")

    # Debug: Check raw image values
    raw_array = np.array(test_image, dtype=np.float32)
    print(
        f"Raw image array - min: {raw_array.min():.6f}, max: {raw_array.max():.6f}, mean: {raw_array.mean():.6f}"
    )

    # Create our custom processor
    custom_processor = Qwen2_5VLImageProcessor(
        patch_size=14,
        temporal_patch_size=2,
        merge_size=2,
    )

    # Process with our custom processor
    print("\nProcessing with our custom processor...")
    custom_result = custom_processor(test_image)

    print("Custom processor output:")
    print(f"  pixel_values shape: {custom_result['pixel_values'].shape}")
    print(f"  image_grid_thw shape: {custom_result['image_grid_thw'].shape}")
    print(
        f"  image_grid_thw values: {custom_result['image_grid_thw'].tolist()}"
    )

    print("\nProcessing with transformers library...")
    transformers_processor = Qwen2VLImageProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        trust_remote_code=True,
    )

    # Let's see what the transformers processor does to the image
    print("Transformers processor config:")
    print(
        f"  image_size: {getattr(transformers_processor, 'image_size', 'N/A')}"
    )
    print(
        f"  patch_size: {getattr(transformers_processor, 'patch_size', 'N/A')}"
    )
    print(
        f"  temporal_patch_size: {getattr(transformers_processor, 'temporal_patch_size', 'N/A')}"
    )

    # Debug: Let's see what happens if we pass the raw image (not preprocessed)
    print("\nTrying with raw image (not preprocessed by fetch_image)...")
    raw_image = Image.open(io.BytesIO(urllib.request.urlopen(image_url).read()))
    print(f"Raw image size: {raw_image.size}")

    transformers_result_raw = transformers_processor(
        raw_image,
        return_tensors="np",
    )

    print("Transformers with raw image:")
    print(
        f"  pixel_values shape: {transformers_result_raw['pixel_values'].shape}"
    )
    print(
        f"  image_grid_thw shape: {transformers_result_raw['image_grid_thw'].shape}"
    )
    print(
        f"  image_grid_thw values: {transformers_result_raw['image_grid_thw'].tolist()}"
    )

    # Now try with our preprocessed image
    transformers_result = transformers_processor(
        test_image,
        return_tensors="np",
    )

    print("Transformers with preprocessed image:")
    print(f"  pixel_values shape: {transformers_result['pixel_values'].shape}")
    print(
        f"  image_grid_thw shape: {transformers_result['image_grid_thw'].shape}"
    )
    print(
        f"  image_grid_thw values: {transformers_result['image_grid_thw'].tolist()}"
    )

    # Calculate expected patches based on grid dimensions
    transformers_grid = transformers_result["image_grid_thw"][0]
    temporal, height_patches, width_patches = transformers_grid
    expected_patches = temporal * height_patches * width_patches
    print(
        f"Expected patches from grid: {expected_patches} (t={temporal}, h={height_patches}, w={width_patches})"
    )

    # ASSERT: Compare shapes
    print("\nComparing outputs:")
    assert (
        custom_result["pixel_values"].shape
        == transformers_result["pixel_values"].shape
    ), (
        f"pixel_values shapes don't match: "
        f"custom {custom_result['pixel_values'].shape} vs "
        f"transformers {transformers_result['pixel_values'].shape}"
    )
    assert (
        custom_result["image_grid_thw"].shape
        == transformers_result["image_grid_thw"].shape
    ), (
        f"image_grid_thw shapes don't match: "
        f"custom {custom_result['image_grid_thw'].shape} vs "
        f"transformers {transformers_result['image_grid_thw'].shape}"
    )
    print("✅ Shapes match")

    # ASSERT: Compare grid dimensions
    assert np.array_equal(
        custom_result["image_grid_thw"],
        transformers_result["image_grid_thw"],
    ), (
        f"Grid dimensions don't match: "
        f"custom {custom_result['image_grid_thw'].tolist()} vs "
        f"transformers {transformers_result['image_grid_thw'].tolist()}"
    )
    print("✅ Grid dimensions match")

    # Check if the difference is systematic
    diff = custom_result["pixel_values"] - transformers_result["pixel_values"]
    print("  Difference (custom - transformers):")
    print(f"    - min: {diff.min():.6f}")
    print(f"    - max: {diff.max():.6f}")
    print(f"    - mean: {diff.mean():.6f}")
    print(f"    - std: {diff.std():.6f}")

    # Check if it's a scaling issue
    ratio = custom_result["pixel_values"] / (
        transformers_result["pixel_values"] + 1e-8
    )
    print("  Ratio (custom / transformers):")
    print(f"    - min: {ratio.min():.6f}")
    print(f"    - max: {ratio.max():.6f}")
    print(f"    - mean: {ratio.mean():.6f}")
    print(f"    - std: {ratio.std():.6f}")

    # Debug: Show first few values from both processors
    print("\nDebug: First 10 values comparison:")
    print(f"  Custom:     {custom_result['pixel_values'][0, :10]}")
    print(f"  Transformers: {transformers_result['pixel_values'][0, :10]}")
    print(f"  Difference: {diff[0, :10]}")

    # Debug: Show values from middle of array
    mid_idx = custom_result["pixel_values"].shape[0] // 2
    print(f"\nDebug: Middle 10 values (row {mid_idx}):")
    print(f"  Custom:     {custom_result['pixel_values'][mid_idx, :10]}")
    print(
        f"  Transformers: {transformers_result['pixel_values'][mid_idx, :10]}"
    )
    print(f"  Difference: {diff[mid_idx, :10]}")

    # ASSERT: Compare pixel values (allow for small numerical differences)
    assert np.allclose(
        custom_result["pixel_values"],
        transformers_result["pixel_values"],
        rtol=1e-5,
        atol=1e-5,
    ), (
        f"Pixel values don't match within tolerance. "
        f"Custom min/max: {custom_result['pixel_values'].min():.6f}/{custom_result['pixel_values'].max():.6f}, "
        f"Transformers min/max: {transformers_result['pixel_values'].min():.6f}/{transformers_result['pixel_values'].max():.6f}"
    )


def test_multiple_images():
    """Test our custom processor with multiple images."""
    # Download multiple images
    image_urls = [
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
        "https://picsum.photos/id/237/200/300",
    ]

    test_images = []
    for url in image_urls:
        image = download_image(url)
        test_images.append(image)

    print(f"\nTesting with {len(test_images)} images")

    # Create our custom processor
    custom_processor = Qwen2_5VLImageProcessor(
        patch_size=14,
        temporal_patch_size=2,
        merge_size=2,
    )

    # Process multiple images with our custom processor
    print("\nProcessing multiple images with our custom processor...")
    custom_result = custom_processor(test_images)

    print("Custom processor output for multiple images:")
    print(f"  pixel_values shape: {custom_result['pixel_values'].shape}")
    print(f"  image_grid_thw shape: {custom_result['image_grid_thw'].shape}")
    print(
        f"  image_grid_thw values: {custom_result['image_grid_thw'].tolist()}"
    )

    # Verify the output makes sense for multiple images
    num_images = len(test_images)
    expected_grid_shape = (
        num_images,
        3,
    )  # (num_images, temporal, height, width)

    print("\nVerifying multiple image output:")
    print(f"  Expected grid shape: {expected_grid_shape}")
    print(f"  Actual grid shape: {custom_result['image_grid_thw'].shape}")

    assert custom_result["image_grid_thw"].shape == expected_grid_shape, (
        f"Grid shape mismatch: expected {expected_grid_shape}, got {custom_result['image_grid_thw'].shape}"
    )

    print("\nProcessing multiple images with transformers library...")
    transformers_processor = Qwen2VLImageProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        trust_remote_code=True,
    )

    transformers_result = transformers_processor(
        test_images,
        return_tensors="np",
    )

    # ASSERT: Compare shapes
    print("\nComparing multiple image outputs:")
    assert (
        custom_result["pixel_values"].shape
        == transformers_result["pixel_values"].shape
    ), (
        f"pixel_values shapes don't match: "
        f"custom {custom_result['pixel_values'].shape} vs "
        f"transformers {transformers_result['pixel_values'].shape}"
    )
    assert (
        custom_result["image_grid_thw"].shape
        == transformers_result["image_grid_thw"].shape
    ), (
        f"image_grid_thw shapes don't match: "
        f"custom {custom_result['image_grid_thw'].shape} vs "
        f"transformers {transformers_result['image_grid_thw'].shape}"
    )
    print("✅ Multiple image shapes match")

    # ASSERT: Compare grid dimensions
    assert np.array_equal(
        custom_result["image_grid_thw"],
        transformers_result["image_grid_thw"],
    ), (
        f"Grid dimensions don't match: "
        f"custom {custom_result['image_grid_thw'].tolist()} vs "
        f"transformers {transformers_result['image_grid_thw'].tolist()}"
    )
    print("✅ Multiple image grid dimensions match")

    # ASSERT: Compare pixel values
    assert np.allclose(
        custom_result["pixel_values"],
        transformers_result["pixel_values"],
        rtol=1e-5,
        atol=1e-5,
    ), (
        f"Pixel values don't match within tolerance. "
        f"Custom min/max: {custom_result['pixel_values'].min():.6f}/{custom_result['pixel_values'].max():.6f}, "
        f"Transformers min/max: {transformers_result['pixel_values'].min():.6f}/{transformers_result['pixel_values'].max():.6f}"
    )
    print("✅ Multiple image pixel values match")
