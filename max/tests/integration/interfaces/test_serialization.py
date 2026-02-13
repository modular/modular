# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Tests for serialization utilities."""

import pickle
from typing import Any

import msgspec
import numpy as np
import numpy.typing as npt
from max.interfaces import RequestID
from max.interfaces.generation import GenerationOutput
from max.interfaces.request.open_responses import OutputImageContent
from max.interfaces.status import GenerationStatus
from max.interfaces.utils.serialization import (
    msgpack_numpy_decoder,
    msgpack_numpy_encoder,
)


class SampleData(msgspec.Struct, tag=True, kw_only=True, omit_defaults=True):
    """Sample data structure with numpy array for testing."""

    array: npt.NDArray[np.integer[Any]]
    value: int
    request_id: RequestID


def test_msgpack_numpy_decoder_pickle_serialization() -> None:
    """Test that MsgpackNumpyDecoder can be pickled and unpickled successfully."""
    # Create test data with a numpy array
    original_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    test_data = SampleData(
        array=original_array, value=42, request_id=RequestID()
    )

    # Encode the test data
    encoder = msgpack_numpy_encoder()
    encoded_data = encoder(test_data)

    # Create a decoder
    original_decoder = msgpack_numpy_decoder(SampleData, copy=True)

    # Test that the original decoder works
    decoded_original = original_decoder(encoded_data)
    assert isinstance(decoded_original, SampleData)
    assert np.array_equal(decoded_original.array, original_array)
    assert decoded_original.value == 42

    # Pickle and unpickle the decoder
    pickled_decoder = pickle.dumps(original_decoder)
    unpickled_decoder = pickle.loads(pickled_decoder)

    # Test that the unpickled decoder still works
    decoded_unpickled = unpickled_decoder(encoded_data)
    assert isinstance(decoded_unpickled, SampleData)
    assert np.array_equal(decoded_unpickled.array, original_array)
    assert decoded_unpickled.value == 42

    # Verify both decoders produce identical results
    assert np.array_equal(decoded_original.array, decoded_unpickled.array)
    assert decoded_original.value == decoded_unpickled.value


def test_msgpack_numpy_decoder_pickle_with_copy_false() -> None:
    """Test pickling decoder with copy=False parameter."""
    # Create test data
    original_array = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    test_data = SampleData(
        array=original_array, value=123, request_id=RequestID()
    )

    # Encode the test data
    encoder = msgpack_numpy_encoder()
    encoded_data = encoder(test_data)

    # Create a decoder with copy=False
    original_decoder = msgpack_numpy_decoder(SampleData, copy=False)

    # Pickle and unpickle the decoder
    pickled_decoder = pickle.dumps(original_decoder)
    unpickled_decoder = pickle.loads(pickled_decoder)

    # Test both decoders
    decoded_original = original_decoder(encoded_data)
    decoded_unpickled = unpickled_decoder(encoded_data)

    # Verify results are correct
    assert isinstance(decoded_original, SampleData)
    assert isinstance(decoded_unpickled, SampleData)
    assert np.array_equal(decoded_original.array, original_array)
    assert np.array_equal(decoded_unpickled.array, original_array)
    assert decoded_original.value == decoded_unpickled.value == 123


def test_msgpack_numpy_decoder_pickle_preserves_parameters() -> None:
    """Test that pickling preserves decoder parameters correctly."""
    # Test different parameter combinations
    test_cases = [
        (SampleData, True),
        (SampleData, False),
    ]

    for type_, copy in test_cases:
        # Create decoder with specific parameters
        original_decoder = msgpack_numpy_decoder(type_, copy=copy)

        # Pickle and unpickle
        pickled_decoder = pickle.dumps(original_decoder)
        unpickled_decoder = pickle.loads(pickled_decoder)

        # Verify internal parameters are preserved
        assert unpickled_decoder._type == type_
        assert unpickled_decoder._copy == copy


def test_msgpack_numpy_encoder_pickle_serialization() -> None:
    """Test that MsgpackNumpyEncoder can be pickled and unpickled successfully."""
    # Create test data with a numpy array
    original_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    test_data = SampleData(
        array=original_array, value=42, request_id=RequestID()
    )

    # Create an encoder
    original_encoder = msgpack_numpy_encoder()

    # Test that the original encoder works
    encoded_original = original_encoder(test_data)
    assert len(encoded_original) > 0

    # Pickle and unpickle the encoder
    pickled_encoder = pickle.dumps(original_encoder)
    unpickled_encoder = pickle.loads(pickled_encoder)

    # Test that the unpickled encoder still works
    encoded_unpickled = unpickled_encoder(test_data)
    assert len(encoded_unpickled) > 0

    # Verify both encoders produce identical results
    assert encoded_original == encoded_unpickled

    # Verify the encoded data can be decoded correctly
    decoder = msgpack_numpy_decoder(SampleData, copy=True)
    decoded_original = decoder(encoded_original)
    decoded_unpickled = decoder(encoded_unpickled)

    assert isinstance(decoded_original, SampleData)
    assert isinstance(decoded_unpickled, SampleData)
    assert np.array_equal(decoded_original.array, original_array)
    assert np.array_equal(decoded_unpickled.array, original_array)
    assert decoded_original.value == decoded_unpickled.value == 42


def test_msgpack_numpy_encoder_pickle_with_shared_memory() -> None:
    """Test pickling encoder with shared memory parameters."""
    # Create test data
    original_array = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    test_data = SampleData(
        array=original_array, value=123, request_id=RequestID()
    )

    # Create an encoder with shared memory enabled
    original_encoder = msgpack_numpy_encoder(
        use_shared_memory=True, shared_memory_threshold=1000
    )

    # Pickle and unpickle the encoder
    pickled_encoder = pickle.dumps(original_encoder)
    unpickled_encoder = pickle.loads(pickled_encoder)

    # Test both encoders
    encoded_original = original_encoder(test_data)
    encoded_unpickled = unpickled_encoder(test_data)

    # Verify results are identical
    assert encoded_original == encoded_unpickled

    # Verify the encoded data can be decoded correctly
    decoder = msgpack_numpy_decoder(SampleData, copy=True)
    decoded_original = decoder(encoded_original)
    decoded_unpickled = decoder(encoded_unpickled)

    assert np.array_equal(decoded_original.array, original_array)
    assert np.array_equal(decoded_unpickled.array, original_array)
    assert decoded_original.value == decoded_unpickled.value == 123


def test_msgpack_numpy_encoder_pickle_preserves_parameters() -> None:
    """Test that pickling preserves encoder parameters correctly."""
    # Test different parameter combinations
    test_cases = [
        (False, 0),
        (True, 1000),
        (False, 5000),
        (True, 0),
    ]

    for use_shared_memory, shared_memory_threshold in test_cases:
        # Create encoder with specific parameters
        original_encoder = msgpack_numpy_encoder(
            use_shared_memory=use_shared_memory,
            shared_memory_threshold=shared_memory_threshold,
        )

        # Pickle and unpickle
        pickled_encoder = pickle.dumps(original_encoder)
        unpickled_encoder = pickle.loads(pickled_encoder)

        # Verify internal parameters are preserved
        assert unpickled_encoder._use_shared_memory == use_shared_memory
        assert (
            unpickled_encoder._shared_memory_threshold
            == shared_memory_threshold
        )


def test_generation_output_serialization() -> None:
    """Test that GenerationOutput can be serialized and deserialized with ZeroMQ."""
    # Create test GenerationOutput with OutputImageContent containing base64 image data
    img_array = np.random.rand(64, 64, 3).astype(np.float32)
    test_output = GenerationOutput(
        request_id=RequestID(),
        final_status=GenerationStatus.END_OF_SEQUENCE,
        output=[
            OutputImageContent.from_numpy(img_array, format="png"),
        ],
    )

    # Encode the GenerationOutput
    encoder = msgpack_numpy_encoder()
    encoded_data = encoder(test_output)
    assert len(encoded_data) > 0

    # Decode the GenerationOutput
    decoder = msgpack_numpy_decoder(GenerationOutput, copy=True)
    decoded_output = decoder(encoded_data)

    # Verify the decoded output matches the original
    assert isinstance(decoded_output, GenerationOutput)
    assert decoded_output.request_id == test_output.request_id
    assert decoded_output.final_status == test_output.final_status
    assert decoded_output.is_done == test_output.is_done
    assert len(decoded_output.output) == len(test_output.output)

    # Verify the OutputImageContent is correct
    assert isinstance(decoded_output.output[0], OutputImageContent)
    assert decoded_output.output[0].type == "output_image"

    # Narrow type for mypy
    decoded_image = decoded_output.output[0]
    test_image = test_output.output[0]
    assert isinstance(decoded_image, OutputImageContent)
    assert isinstance(test_image, OutputImageContent)
    assert decoded_image.image_data == test_image.image_data
    assert decoded_image.format == "png"


def test_generation_output_serialization_with_multiple_images() -> None:
    """Test serialization of GenerationOutput with multiple images."""
    # Create test GenerationOutput with multiple OutputImageContent objects
    img_array1 = np.random.rand(32, 32, 3).astype(np.float32)
    img_array2 = np.random.rand(64, 64, 3).astype(np.float32)

    test_output = GenerationOutput(
        request_id=RequestID(),
        final_status=GenerationStatus.END_OF_SEQUENCE,
        output=[
            OutputImageContent.from_numpy(img_array1, format="png"),
            OutputImageContent.from_numpy(img_array2, format="jpeg"),
        ],
    )

    # Encode and decode
    encoder = msgpack_numpy_encoder()
    encoded_data = encoder(test_output)

    decoder = msgpack_numpy_decoder(GenerationOutput, copy=True)
    decoded_output = decoder(encoded_data)

    # Verify all images are preserved
    assert len(decoded_output.output) == 2

    # Narrow types for mypy
    decoded_image_0 = decoded_output.output[0]
    decoded_image_1 = decoded_output.output[1]
    test_image_0 = test_output.output[0]
    test_image_1 = test_output.output[1]
    assert isinstance(decoded_image_0, OutputImageContent)
    assert isinstance(decoded_image_1, OutputImageContent)
    assert isinstance(test_image_0, OutputImageContent)
    assert isinstance(test_image_1, OutputImageContent)

    assert decoded_image_0.format == "png"
    assert decoded_image_1.format == "jpeg"
    assert decoded_image_0.image_data == test_image_0.image_data
    assert decoded_image_1.image_data == test_image_1.image_data
