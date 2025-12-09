# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


"""Simple test to compare Qwen3VLImageProcessor with transformers library using real images."""

from __future__ import annotations

import io
import urllib.request

import numpy as np

# Import our custom processor
from max.pipelines.architectures.qwen3vl_moe.tokenizer import (
    Qwen3VLImageProcessor,
)
from PIL import Image
from transformers import Qwen3VLProcessor


def download_image(url: str) -> Image.Image:
    """Download an image from URL and apply the same preprocessing as transformers library."""
    print(f"Downloading image from: {url}")
    # First download the raw image
    with urllib.request.urlopen(url) as response:
        image_data = response.read()
    raw_image = Image.open(io.BytesIO(image_data))

    print(f"Raw image size: {raw_image.size}")
    return raw_image


def test_image_processor() -> None:
    """Compare our custom processor with transformers library."""
    # Download the demo image
    image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    test_image = download_image(image_url)

    print(f"Original image size: {test_image.size}")
    print(f"Original image mode: {test_image.mode}")

    # Debug: Check raw image values
    raw_array = np.array(test_image, dtype=np.float32)
    print(
        f"Raw image array - min: {raw_array.min():.6f}, max:"
        f" {raw_array.max():.6f}, mean: {raw_array.mean():.6f}"
    )

    # Create our custom processor
    custom_processor = Qwen3VLImageProcessor(
        patch_size=16,
        temporal_patch_size=2,
        merge_size=2,
    )

    # Process with our custom processor
    print("\nProcessing with our custom processor...")
    pixel_values, image_grid_thw, pixel_values_list = custom_processor(
        test_image
    )

    print("Custom processor output:")
    print(f"  concatenated_pixel_values shape: {pixel_values.shape}")
    print(f"  image_grid_thw shape: {image_grid_thw.shape}")
    print(f"  image_grid_thw values: {image_grid_thw.tolist()}")
    print(f"  pixel_values list length: {len(pixel_values_list)}")

    print("\nProcessing with transformers library...")
    transformers_processor = Qwen3VLProcessor.from_pretrained(
        "Qwen/Qwen3-VL-30B-A3B-Instruct",
        trust_remote_code=True,
    )

    # Use the image processor directly from the transformers processor
    transformers_image_processor = transformers_processor.image_processor

    # Let's see what the transformers image processor does to the image
    print("Transformers image processor config:")
    print(
        "  patch_size:"
        f" {getattr(transformers_image_processor, 'patch_size', 'N/A')}"
    )
    print(
        "  temporal_patch_size:"
        f" {getattr(transformers_image_processor, 'temporal_patch_size', 'N/A')}"
    )

    # Debug: Let's see what happens if we pass the raw image (not preprocessed)
    print("\nTrying with raw image (not preprocessed by fetch_image)...")

    transformers_result_raw = transformers_image_processor(
        images=[image_url],
        return_tensors="pt",
    )
    # Convert PyTorch tensors to NumPy for comparison
    transformers_result_raw = {
        key: value.numpy() if hasattr(value, "numpy") else value
        for key, value in transformers_result_raw.items()
    }

    print("Transformers with raw image:")
    print(
        f"  pixel_values shape: {transformers_result_raw['pixel_values'].shape}"
    )
    print(
        "  image_grid_thw shape:"
        f" {transformers_result_raw['image_grid_thw'].shape}"
    )
    print(
        "  image_grid_thw values:"
        f" {transformers_result_raw['image_grid_thw'].tolist()}"
    )

    # Calculate expected patches based on grid dimensions
    transformers_grid = transformers_result_raw["image_grid_thw"][0]
    temporal, height_patches, width_patches = transformers_grid
    expected_patches = temporal * height_patches * width_patches
    print(
        f"Expected patches from grid: {expected_patches} (t={temporal},"
        f" h={height_patches}, w={width_patches})"
    )

    # ASSERT: Compare shapes
    print("\nComparing outputs:")
    assert (
        pixel_values.shape == transformers_result_raw["pixel_values"].shape
    ), (
        "pixel_values shapes don't match: "
        f"custom {pixel_values.shape} vs "
        f"transformers {transformers_result_raw['pixel_values'].shape}"
    )
    assert (
        image_grid_thw.shape == transformers_result_raw["image_grid_thw"].shape
    ), (
        "image_grid_thw shapes don't match: "
        f"custom {image_grid_thw.shape} vs "
        f"transformers {transformers_result_raw['image_grid_thw'].shape}"
    )
    print("✅ Shapes match")

    # ASSERT: Compare grid dimensions
    assert np.array_equal(
        image_grid_thw,
        transformers_result_raw["image_grid_thw"],
    ), (
        "Grid dimensions don't match: "
        f"custom {image_grid_thw.tolist()} vs "
        f"transformers {transformers_result_raw['image_grid_thw'].tolist()}"
    )
    print("✅ Grid dimensions match")

    # Check if the difference is systematic
    diff = pixel_values - transformers_result_raw["pixel_values"]
    print("  Difference (custom - transformers):")
    print(f"    - min: {diff.min():.6f}")
    print(f"    - max: {diff.max():.6f}")
    print(f"    - mean: {diff.mean():.6f}")
    print(f"    - std: {diff.std():.6f}")

    # Check if it's a scaling issue
    ratio = pixel_values / (transformers_result_raw["pixel_values"] + 1e-8)
    print("  Ratio (custom / transformers):")
    print(f"    - min: {ratio.min():.6f}")
    print(f"    - max: {ratio.max():.6f}")
    print(f"    - mean: {ratio.mean():.6f}")
    print(f"    - std: {ratio.std():.6f}")

    # Debug: Show first few values from both processors
    print("\nDebug: First 10 values comparison:")
    print(f"  Custom:     {pixel_values[0, :10]}")
    print(f"  Transformers: {transformers_result_raw['pixel_values'][0, :10]}")
    print(f"  Difference: {diff[0, :10]}")

    # Debug: Show values from middle of array
    mid_idx = pixel_values.shape[0] // 2
    print(f"\nDebug: Middle 10 values (row {mid_idx}):")
    print(f"  Custom:     {pixel_values[mid_idx, :10]}")
    print(
        "  Transformers:"
        f" {transformers_result_raw['pixel_values'][mid_idx, :10]}"
    )
    print(f"  Difference: {diff[mid_idx, :10]}")

    # ASSERT: Compare pixel values (print max abs/rel errors)
    abs_err = np.max(
        np.abs(pixel_values - transformers_result_raw["pixel_values"])
    )
    rel_err = np.max(
        np.abs(pixel_values - transformers_result_raw["pixel_values"])
        / (np.abs(transformers_result_raw["pixel_values"]) + 1e-12)
    )
    print(
        f"Max absolute error: {abs_err:.8f}, Max relative error: {rel_err:.8f}"
    )

    assert np.allclose(
        pixel_values,
        transformers_result_raw["pixel_values"],
        rtol=1e-2,
        atol=1e-2,
    ), (
        "Pixel values don't match within tolerance. "
        f"Max abs err: {abs_err:.8f}, Max rel err: {rel_err:.8f}. "
        f"Custom min/max: {pixel_values.min():.6f}/{pixel_values.max():.6f}, "
        "Transformers min/max:"
        f" {transformers_result_raw['pixel_values'].min():.6f}/{transformers_result_raw['pixel_values'].max():.6f}"
    )
    print("✅ Pixel values match")


def test_multiple_images() -> None:
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
    custom_processor = Qwen3VLImageProcessor(
        patch_size=16,
        temporal_patch_size=2,
        merge_size=2,
    )

    # Process multiple images with our custom processor
    print("\nProcessing multiple images with our custom processor...")
    pixel_values, image_grid_thw, _ = custom_processor(test_images)

    print("Custom processor output for multiple images:")
    print(f"  concatenated_pixel_values shape: {pixel_values.shape}")
    print(f"  image_grid_thw shape: {image_grid_thw.shape}")
    print(f"  image_grid_thw values: {image_grid_thw.tolist()}")

    # Verify the output makes sense for multiple images
    num_images = len(test_images)
    expected_grid_shape = (
        num_images,
        3,
    )  # (num_images, temporal, height, width)

    print("\nVerifying multiple image output:")
    print(f"  Expected grid shape: {expected_grid_shape}")
    print(f"  Actual grid shape: {image_grid_thw.shape}")

    assert image_grid_thw.shape == expected_grid_shape, (
        f"Grid shape mismatch: expected {expected_grid_shape}, got"
        f" {image_grid_thw.shape}"
    )

    print("\nProcessing multiple images with transformers library...")
    transformers_processor = Qwen3VLProcessor.from_pretrained(
        "Qwen/Qwen3-VL-30B-A3B-Instruct",
        trust_remote_code=True,
    )

    # Use the image processor directly from the transformers processor
    transformers_image_processor = transformers_processor.image_processor
    transformers_result = transformers_image_processor(
        images=image_urls,
        return_tensors="pt",
    )
    # Convert PyTorch tensors to NumPy for comparison
    transformers_result = {
        key: value.numpy() if hasattr(value, "numpy") else value
        for key, value in transformers_result.items()
    }

    print("Transformers result:")
    print(f"  pixel_values shape: {transformers_result['pixel_values'].shape}")
    print(
        f"  image_grid_thw shape: {transformers_result['image_grid_thw'].shape}"
    )
    print(
        "  image_grid_thw values:"
        f" {transformers_result['image_grid_thw'].tolist()}"
    )

    # ASSERT: Compare shapes
    print("\nComparing multiple image outputs:")
    assert pixel_values.shape == transformers_result["pixel_values"].shape, (
        "pixel_values shapes don't match: "
        f"custom {pixel_values.shape} vs "
        f"transformers {transformers_result['pixel_values'].shape}"
    )
    assert (
        image_grid_thw.shape == transformers_result["image_grid_thw"].shape
    ), (
        "image_grid_thw shapes don't match: "
        f"custom {image_grid_thw.shape} vs "
        f"transformers {transformers_result['image_grid_thw'].shape}"
    )
    print("✅ Multiple image shapes match")

    # ASSERT: Compare grid dimensions
    assert np.array_equal(
        image_grid_thw,
        transformers_result["image_grid_thw"],
    ), (
        "Grid dimensions don't match: "
        f"custom {image_grid_thw.tolist()} vs "
        f"transformers {transformers_result['image_grid_thw'].tolist()}"
    )
    print("✅ Multiple image grid dimensions match")

    # ASSERT: Compare pixel values (print max abs/rel errors)
    abs_err_multi = np.max(
        np.abs(pixel_values - transformers_result["pixel_values"])
    )
    rel_err_multi = np.max(
        np.abs(pixel_values - transformers_result["pixel_values"])
        / (np.abs(transformers_result["pixel_values"]) + 1e-12)
    )
    print(
        f"Max absolute error (multi): {abs_err_multi:.8f}, Max relative error"
        f" (multi): {rel_err_multi:.8f}"
    )

    assert np.allclose(
        pixel_values,
        transformers_result["pixel_values"],
        rtol=1e-2,
        atol=1e-2,
    ), (
        "Pixel values don't match within tolerance. "
        f"Max abs err: {abs_err_multi:.8f}, Max rel err: {rel_err_multi:.8f}. "
        f"Custom min/max: {pixel_values.min():.6f}/{pixel_values.max():.6f}, "
        "Transformers min/max:"
        f" {transformers_result['pixel_values'].min():.6f}/{transformers_result['pixel_values'].max():.6f}"
    )
    print("✅ Multiple image pixel values match")
