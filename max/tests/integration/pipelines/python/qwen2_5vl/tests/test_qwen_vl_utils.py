# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Test qwen_vl_utils against original implementation."""

import base64
import tempfile
from collections.abc import Generator
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pytest
import qwen_vl_utils.vision_process as original_utils
from max.pipelines.architectures.qwen2_5vl.nn.qwen_vl_utils import (
    extract_vision_info,
    fetch_image,
    process_vision_info,
    smart_resize,
)
from PIL import Image


def create_test_image(width: int = 224, height: int = 224) -> Image.Image:
    """Create a test image for testing."""
    img_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def compare_images_numerically(img1: Image.Image, img2: Image.Image) -> None:
    """Compare two PIL Images numerically with a tolerance for small differences."""
    # Convert to numpy arrays for numerical comparison
    arr1 = np.array(img1)
    arr2 = np.array(img2)

    # Check shapes match
    assert arr1.shape == arr2.shape, (
        f"Image shapes do not match: {arr1.shape} != {arr2.shape}"
    )

    # Calculate absolute difference
    np.testing.assert_array_equal(arr1, arr2)


def create_test_video_file() -> str:
    """Create a simple test video file for testing."""
    # Create a temporary video file
    temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_path = temp_file.name
    temp_file.close()

    # Create a simple test video with OpenCV
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_path, fourcc, 2.0, (224, 224))

    # Create 10 frames with different colors
    for i in range(10):
        # Create a frame with a solid color that changes
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        frame[:, :] = [i * 25 % 255, (i * 50) % 255, (i * 75) % 255]
        out.write(frame)

    out.release()
    return temp_path


@pytest.fixture
def test_image() -> Image.Image:
    """Fixture providing a test image."""
    return create_test_image()


@pytest.fixture
def test_image_base64(test_image: Image.Image) -> str:
    """Fixture providing a test image as base64 string."""
    return image_to_base64(test_image)


@pytest.fixture
def test_image_file(test_image: Image.Image) -> Generator[str, None, None]:
    """Fixture providing a test image file path."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        test_image.save(f.name)
        yield f.name
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def test_video_file() -> Generator[str, None, None]:
    """Fixture providing a test video file path."""
    video_path = create_test_video_file()
    yield video_path
    Path(video_path).unlink(missing_ok=True)


class TestOriginalQwenVLUtils:
    """Test against original qwen_vl_utils implementation."""

    def test_process_vision_info_comparison_images_only(
        self, test_image: Image.Image, test_image_base64: str
    ) -> None:
        """Compare process_vision_info outputs between original and MAX implementation for images."""
        conversations = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image", "image": test_image},
                    {"type": "image", "image": test_image_base64},
                ],
            }
        ]

        # Test MAX implementation
        max_result = process_vision_info(
            conversations, return_video_kwargs=False
        )
        max_images, max_videos, max_kwargs = max_result

        # Test original implementation
        original_result = original_utils.process_vision_info(conversations)
        orig_images, orig_videos = original_result  # type: ignore

        # Compare structure
        assert len(max_result) == 3  # MAX version always returns 3 values
        assert (
            len(original_result) == 2
        )  # Original returns 2 when return_video_kwargs=False

        # Compare image outputs
        assert max_images is not None
        assert orig_images is not None
        assert len(max_images) == len(orig_images)

        for i, (max_img, orig_img) in enumerate(zip(max_images, orig_images)):
            assert isinstance(max_img, Image.Image)
            assert isinstance(orig_img, Image.Image)
            assert max_img.mode == orig_img.mode
            # Sizes should be the same since smart_resize should behave identically
            assert max_img.size == orig_img.size, (
                f"Image {i}: max_size={max_img.size}, orig_size={orig_img.size}"
            )

            # Numerical comparison of pixel values
            compare_images_numerically(max_img, orig_img)

        # Both should return None for videos when no videos are present
        assert max_videos is None
        assert orig_videos is None

        # MAX implementation always returns kwargs (None when return_video_kwargs=False)
        assert max_kwargs is None

    def test_process_vision_info_comparison_with_video_kwargs(
        self, test_image: Image.Image
    ) -> None:
        """Compare process_vision_info outputs with return_video_kwargs=True."""
        conversations = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": test_image},
                ],
            }
        ]

        # Test MAX implementation
        max_result = process_vision_info(
            conversations, return_video_kwargs=True
        )
        max_images, max_videos, max_kwargs = max_result

        # Test original implementation
        original_result = original_utils.process_vision_info(
            conversations, return_video_kwargs=True
        )
        orig_images, orig_videos, orig_kwargs = original_result

        # Both should return 3 values
        assert len(max_result) == 3
        assert len(original_result) == 3

        # Compare images
        assert max_images is not None
        assert orig_images is not None
        assert len(max_images) == len(orig_images)

        for max_img, orig_img in zip(max_images, orig_images):
            assert isinstance(max_img, Image.Image)
            assert isinstance(orig_img, Image.Image)
            assert max_img.mode == orig_img.mode
            assert max_img.size == orig_img.size

            compare_images_numerically(max_img, orig_img)
        # Both should return None for videos
        assert max_videos is None
        assert orig_videos is None

        # Both should return kwargs (empty dict when no videos)
        assert max_kwargs is not None
        assert orig_kwargs is not None
        # Both should be empty or have same structure
        assert isinstance(max_kwargs, dict)
        assert isinstance(orig_kwargs, dict)

    def test_process_vision_info_comparison_no_vision(self) -> None:
        """Compare process_vision_info outputs when no vision content is present."""
        conversations = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello, how are you?"}],
            }
        ]

        # Test MAX implementation
        max_result = process_vision_info(
            conversations, return_video_kwargs=True
        )
        max_images, max_videos, max_kwargs = max_result

        # Test original implementation
        original_result = original_utils.process_vision_info(
            conversations, return_video_kwargs=True
        )
        orig_images, orig_videos, orig_kwargs = original_result

        # Both should return None for images and videos
        assert max_images is None
        assert orig_images is None
        assert max_videos is None
        assert orig_videos is None

        # Both should return kwargs
        assert max_kwargs is not None
        assert orig_kwargs is not None

    def test_process_vision_info_comparison_mixed_formats(
        self,
        test_image: Image.Image,
        test_image_base64: str,
        test_image_file: Generator[str, None, None],
    ) -> None:
        """Compare process_vision_info with multiple image formats."""
        conversations = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": test_image},  # PIL Image
                    {"type": "image", "image": test_image_base64},  # Base64
                    {"type": "image", "image": test_image_file},  # File path
                ],
            }
        ]

        # Test MAX implementation
        max_result = process_vision_info(
            conversations, return_video_kwargs=False
        )
        max_images, max_videos, max_kwargs = max_result

        # Test original implementation
        original_result = original_utils.process_vision_info(conversations)
        orig_images, orig_videos = original_result  # type: ignore

        # Should handle all three image formats
        assert max_images is not None
        assert orig_images is not None
        assert len(max_images) == 3
        assert len(orig_images) == 3

        # Check each image
        for i, (max_img, orig_img) in enumerate(zip(max_images, orig_images)):
            assert isinstance(max_img, Image.Image), (
                f"MAX image {i} is not PIL Image"
            )
            assert isinstance(orig_img, Image.Image), (
                f"Original image {i} is not PIL Image"
            )
            assert max_img.mode == "RGB", (
                f"MAX image {i} mode is {max_img.mode}, expected RGB"
            )
            assert orig_img.mode == "RGB", (
                f"Original image {i} mode is {orig_img.mode}, expected RGB"
            )

            # Sizes should be the same (both use smart_resize)
            assert max_img.size == orig_img.size, (
                f"Image {i} size mismatch: ours={max_img.size}, orig={orig_img.size}"
            )

            # Both should be divisible by IMAGE_FACTOR (28)
            assert max_img.size[0] % 28 == 0, (
                f"MAX image {i} width not divisible by 28"
            )
            assert max_img.size[1] % 28 == 0, (
                f"MAX image {i} height not divisible by 28"
            )
            assert orig_img.size[0] % 28 == 0, (
                f"Original image {i} width not divisible by 28"
            )
            assert orig_img.size[1] % 28 == 0, (
                f"Original image {i} height not divisible by 28"
            )

            # Numerical comparison for each format
            compare_images_numerically(max_img, orig_img)

    def test_process_vision_info_comparison_with_video(
        self, test_video_file: str, test_image: Image.Image
    ) -> None:
        """Compare process_vision_info outputs with video content."""
        conversations = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this video and image?"},
                    {"type": "video", "video": test_video_file},
                    {"type": "image", "image": test_image},
                ],
            }
        ]

        # Test original implementation first to see expected behavior
        original_result = original_utils.process_vision_info(
            conversations, return_video_kwargs=True
        )
        orig_images, orig_videos, orig_kwargs = original_result

        # Test MAX implementation
        with pytest.raises(ImportError, match="No video reader backend"):
            max_result = process_vision_info(
                conversations, return_video_kwargs=True
            )
            max_images, max_videos, max_kwargs = max_result

        if False:  # Disabled because we don't support processing videos yet.
            # Compare images (should be identical)
            if orig_images is not None and max_images is not None:
                assert len(max_images) == len(orig_images)
                for max_img, orig_img in zip(max_images, orig_images):
                    assert isinstance(max_img, Image.Image)
                    assert isinstance(orig_img, Image.Image)
                    compare_images_numerically(max_img, orig_img)

            # Compare video outputs
            if orig_videos is not None and max_videos is not None:
                assert len(max_videos) == len(orig_videos)
                # For videos, the comparison depends on the backend
                # Since we might have different backends, we'll compare shapes and types
                for i, (max_vid, orig_vid) in enumerate(
                    zip(max_videos, orig_videos)
                ):
                    # Check if both are numpy arrays or lists of PIL Images
                    assert type(max_vid) is type(orig_vid), (
                        f"Video {i} type mismatch: MAX={type(max_vid)}, orig={type(orig_vid)}"
                    )

                    if isinstance(max_vid, np.ndarray) and isinstance(
                        orig_vid, np.ndarray
                    ):
                        # Both are numpy arrays (from video backends)
                        assert max_vid.shape == orig_vid.shape, (
                            f"Video {i} shape mismatch: MAX={max_vid.shape}, orig={orig_vid.shape}"
                        )
                        # Note: Exact pixel comparison might fail due to different video backends
                        # so we only compare shapes and data types
                        assert max_vid.dtype == orig_vid.dtype
                    elif isinstance(max_vid, list) and isinstance(
                        orig_vid, list
                    ):
                        # Both are lists of PIL Images
                        assert len(max_vid) == len(orig_vid)
                        for max_frame, orig_frame in zip(max_vid, orig_vid):
                            assert isinstance(max_frame, Image.Image)
                            assert isinstance(orig_frame, Image.Image)
                            assert max_frame.size == orig_frame.size

            # Compare video kwargs
            if orig_kwargs is not None and max_kwargs is not None:
                assert isinstance(max_kwargs, dict)
                assert isinstance(orig_kwargs, dict)
                # Both should have fps information if videos were processed
                if "fps" in orig_kwargs:
                    assert "fps" in max_kwargs
                    # FPS lists should have same length
                    assert len(max_kwargs["fps"]) == len(orig_kwargs["fps"])

    def test_video_only_processing_comparison(
        self, test_video_file: str
    ) -> None:
        """Test video-only processing between implementations."""
        conversations = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this video."},
                    {
                        "type": "video",
                        "video": test_video_file,
                        "fps": 1.0,
                        "max_frames": 8,
                    },
                ],
            }
        ]

        # Test if either implementation can handle video-only content
        original_result = original_utils.process_vision_info(
            conversations, return_video_kwargs=True
        )
        orig_images, orig_videos, orig_kwargs = original_result

        with pytest.raises(ImportError, match="No video reader backend"):
            max_result = process_vision_info(
                conversations, return_video_kwargs=True
            )
            max_images, max_videos, max_kwargs = max_result

    def test_smart_resize_comparison(self) -> None:
        """Compare smart_resize outputs between implementations."""
        test_cases = [
            (100, 100),
            (200, 150),
            (300, 400),
            (50, 200),
            (224, 224),
            (512, 256),
        ]

        for height, width in test_cases:
            max_result = smart_resize(height, width)
            orig_result = original_utils.smart_resize(height, width)

            assert max_result == orig_result, (
                f"smart_resize({height}, {width}): "
                f"max_result={max_result}, orig_result={orig_result}"
            )

    def test_extract_vision_info_comparison(
        self, test_image: Image.Image
    ) -> None:
        """Compare extract_vision_info outputs between implementations."""
        conversations: list[dict] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image", "image": test_image},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "I can see an image."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": test_image},
                    {"type": "text", "text": "And another image?"},
                ],
            },
        ]

        max_result = extract_vision_info(conversations)
        orig_result = original_utils.extract_vision_info(conversations)

        # Should extract the same number of vision elements
        assert len(max_result) == len(orig_result)

        # Should extract the same types of elements
        for max_item, orig_item in zip(max_result, orig_result):
            assert set(max_item.keys()) == set(orig_item.keys())

            # Both should identify the same vision content types
            if "image" in max_item:
                assert "image" in orig_item
            if "image_url" in max_item:
                assert "image_url" in orig_item
            if "video" in max_item:
                assert "video" in orig_item

    def test_extract_vision_info_with_video(
        self, test_image: Image.Image, test_video_file: str
    ) -> None:
        """Test extract_vision_info with mixed image and video content."""
        conversations = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this content:"},
                    {"type": "image", "image": test_image},
                    {"type": "video", "video": test_video_file},
                    {"type": "video", "video": test_video_file, "fps": 2.0},
                ],
            }
        ]

        max_result = extract_vision_info(conversations)
        orig_result = original_utils.extract_vision_info(conversations)

        # Both should extract the same number of vision elements
        assert len(max_result) == len(orig_result)
        assert len(max_result) == 3  # 1 image + 2 videos

        # Check that the right types are extracted
        vision_types_max = [item["type"] for item in max_result if item]
        vision_types_orig = [item["type"] for item in orig_result if item]

        assert vision_types_max == vision_types_orig
        assert "image" in vision_types_max
        assert vision_types_max.count("video") == 2

    def test_fetch_image_numerical_comparison(
        self,
        test_image: Image.Image,
        test_image_base64: str,
        test_image_file: Generator[str, None, None],
    ) -> None:
        """Compare fetch_image outputs numerically between implementations."""
        test_cases = [
            {"name": "PIL Image", "ele": {"image": test_image}},
            {"name": "Base64", "ele": {"image": test_image_base64}},
            {"name": "File path", "ele": {"image": test_image_file}},
            {"name": "File URL", "ele": {"image": f"file://{test_image_file}"}},
        ]

        for case in test_cases:
            print(f"Testing {case['name']} format...")

            # Get results from both implementations
            max_result = fetch_image(case["ele"])  # type: ignore
            orig_result = original_utils.fetch_image(case["ele"])  # type: ignore

            # Basic checks
            assert isinstance(max_result, Image.Image)
            assert isinstance(orig_result, Image.Image)
            assert max_result.mode == orig_result.mode
            assert max_result.size == orig_result.size

            # Numerical comparison
            compare_images_numerically(max_result, orig_result)

    @pytest.mark.parametrize(
        "ele",
        [
            {
                "resized_height": 224,
                "resized_width": 336,
            },
            {
                "min_pixels": 30000,
                "max_pixels": 80000,
            },
            {
                "resized_height": 448,
                "resized_width": 448,
            },
        ],
    )
    def test_image_processing_consistency_with_parameters(
        self, test_image: Image.Image, ele: dict
    ) -> None:
        """Test that image processing with custom parameters produces identical results."""
        from max.pipelines.architectures.qwen2_5vl.nn.qwen_vl_utils import (
            fetch_image as max_fetch_image,
        )

        ele["image"] = test_image

        max_result = max_fetch_image(ele)
        orig_result = original_utils.fetch_image(ele)

        # Size should match exactly
        assert max_result.size == orig_result.size, (
            f"Size mismatch - MAX={max_result.size}, orig={orig_result.size}"
        )

        # Numerical comparison should be very close
        compare_images_numerically(max_result, orig_result)
