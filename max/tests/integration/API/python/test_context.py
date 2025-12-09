# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pickle
import re

import numpy as np
import pytest
from max.interfaces import (
    GenerationStatus,
    ImageMetadata,
    RequestID,
    SamplingParams,
    SamplingParamsGenerationConfigDefaults,
    SamplingParamsInput,
    TextGenerationContext,
    VLMTextGenerationContext,
    msgpack_eq,
    msgpack_numpy_decoder,
    msgpack_numpy_encoder,
)
from max.pipelines.core import TextAndVisionContext, TextContext, TTSContext


def test_context__get_min_token_logit_mask() -> None:
    context = TextContext(
        request_id=RequestID(),
        max_length=10,
        tokens=np.array([0, 1, 2, 3]),
        eos_token_ids={4},
        sampling_params=SamplingParams(min_new_tokens=3),
    )
    vocab_mask = context.get_min_token_logit_mask(1)
    assert len(vocab_mask) == 1
    assert vocab_mask[0].tolist() == [[0, 4]]

    context.update(1)
    vocab_mask = context.get_min_token_logit_mask(1)
    assert len(vocab_mask) == 1
    assert vocab_mask[0].tolist() == [[0, 4]]

    context.update(2)
    vocab_mask = context.get_min_token_logit_mask(3)
    assert len(vocab_mask) == 3
    assert vocab_mask[0].tolist() == [[0, 4]]
    assert vocab_mask[1].tolist() == []
    assert vocab_mask[2].tolist() == []


def test_context__get_min_token_logit_mask_with_multiple_eos_token_ids() -> (
    None
):
    context = TextContext(
        request_id=RequestID(),
        max_length=10,
        tokens=np.array([0, 1, 2, 3]),
        sampling_params=SamplingParams(min_new_tokens=3),
        eos_token_ids={4, 5},
    )
    vocab_mask = context.get_min_token_logit_mask(1)
    assert len(vocab_mask) == 1
    assert vocab_mask[0].tolist() == [[0, 4], [0, 5]]

    context.update(1)
    vocab_mask = context.get_min_token_logit_mask(1)
    assert len(vocab_mask) == 1
    assert vocab_mask[0].tolist() == [[0, 4], [0, 5]]

    context.update(2)
    vocab_mask = context.get_min_token_logit_mask(3)
    assert len(vocab_mask) == 3
    assert vocab_mask[0].tolist() == [[0, 4], [0, 5]]
    assert vocab_mask[1].tolist() == []
    assert vocab_mask[2].tolist() == []


def test_context__get_min_token_logit_mask_with_multiple_eos_token_ids_multistep() -> (
    None
):
    context = TextContext(
        request_id=RequestID(),
        max_length=10,
        tokens=np.array([0, 1, 2, 3]),
        sampling_params=SamplingParams(min_new_tokens=3),
        eos_token_ids={4, 5},
    )
    vocab_mask = context.get_min_token_logit_mask(4)
    assert len(vocab_mask) == 4
    assert vocab_mask[0].tolist() == [[0, 4], [0, 5]]
    assert vocab_mask[1].tolist() == [[0, 4], [0, 5]]
    assert vocab_mask[2].tolist() == [[0, 4], [0, 5]]
    assert vocab_mask[3].tolist() == []

    context.update(1)
    context.update(1)
    context.update(1)
    context.update(1)
    vocab_mask = context.get_min_token_logit_mask(1)
    assert len(vocab_mask) == 1
    assert vocab_mask[0].tolist() == []


def test_context__get_min_token_logit_mask_with_no_eos_token_ids() -> None:
    context = TextContext(
        request_id=RequestID(),
        max_length=10,
        tokens=np.array([0, 1, 2, 3]),
        sampling_params=SamplingParams(min_new_tokens=3),
    )
    vocab_mask = context.get_min_token_logit_mask(1)
    assert len(vocab_mask) == 1
    assert vocab_mask[0].tolist() == []

    context.update(1)
    vocab_mask = context.get_min_token_logit_mask(1)
    assert len(vocab_mask) == 1
    assert vocab_mask[0].tolist() == []

    context.update(2)
    vocab_mask = context.get_min_token_logit_mask(3)
    assert len(vocab_mask) == 3
    assert vocab_mask[0].tolist() == []
    assert vocab_mask[1].tolist() == []
    assert vocab_mask[2].tolist() == []


def test_context__get_min_token_logit_mask_with_no_min_new_tokens() -> None:
    context = TextContext(
        request_id=RequestID(),
        max_length=10,
        tokens=np.array([0, 1, 2, 3]),
        eos_token_ids={4, 5},
    )
    vocab_mask = context.get_min_token_logit_mask(1)
    assert len(vocab_mask) == 1
    assert vocab_mask[0].tolist() == []

    context.update(1)
    vocab_mask = context.get_min_token_logit_mask(1)
    assert len(vocab_mask) == 1
    assert vocab_mask[0].tolist() == []

    context.update(2)
    vocab_mask = context.get_min_token_logit_mask(3)
    assert len(vocab_mask) == 3
    assert vocab_mask[0].tolist() == []
    assert vocab_mask[1].tolist() == []
    assert vocab_mask[2].tolist() == []


def test_context__eos() -> None:
    context = TextContext(
        request_id=RequestID(),
        max_length=10,
        tokens=np.array([0, 1, 2, 3]),
        eos_token_ids={4},
    )
    assert context.eos_token_ids == {4}
    assert context.is_initial_prompt
    context.update(4)
    assert not context.is_initial_prompt
    assert context.current_length == 5
    assert context.status == GenerationStatus.END_OF_SEQUENCE


def test_context__max_length() -> None:
    context = TextContext(
        request_id=RequestID(),
        max_length=6,
        tokens=np.array([0, 1, 2, 3]),
    )
    for i in range(2):
        assert context.status == GenerationStatus.ACTIVE
        context.update(i)
    assert context.status == GenerationStatus.MAXIMUM_LENGTH


def test_context__current_length() -> None:
    context = TextContext(
        request_id=RequestID(),
        max_length=10,
        tokens=np.array([0, 1, 2, 3]),
    )

    assert context.current_length == 4
    assert context.is_initial_prompt

    context.update(4)
    assert not context.is_initial_prompt
    assert context.current_length == 5

    # Currently, there are 5 tokens, we are saying
    # here is the next one, and we've generated 3 tokens
    # including that one, so increment the current length
    # accordingly.
    for i in range(3):
        context.update(5 + i)

    assert context.current_length == 8


def test_context__seq_len() -> None:
    context = TextContext(
        request_id=RequestID(),
        max_length=10,
        tokens=np.array([0, 1, 2, 3]),
    )

    assert context.active_length == 4
    context.update(4)
    assert context.active_length == 1
    for i in range(5):
        context.update(5 + i)
    assert context.active_length == 1


def test_context__needs_ce() -> None:
    context = TextContext(
        request_id=RequestID(),
        max_length=10,
        tokens=np.array([0, 1, 2, 3]),
    )

    # There are 4 unencoded prompt tokens
    assert context.active_length == 4
    assert context.needs_ce is True

    # Encode 2/4 prompt tokens
    context.chunk(2)
    assert context.active_length == 2
    assert context.needs_ce
    context.update(98)  # token 98 is discarded
    assert context.all_tokens.tolist() == [0, 1, 2, 3]

    # There are 2 unencoded prompt tokens left
    assert context.needs_ce is True
    assert context.active_length == 2

    # Create a bunch of draft tokens like in spec decoding
    context.update(99)
    context.update(100)
    context.update(101)
    context.update(102)
    assert context.all_tokens.tolist() == [0, 1, 2, 3, 99, 100, 101, 102]
    context.rewind_processing(2)

    # Even though the active length is 3, we are not in CE mode!
    assert context.next_tokens.tolist() == [100, 101, 102]
    assert context.active_length == 3
    assert context.needs_ce is False


def test_context__skip_processing() -> None:
    context = TextContext(
        request_id=RequestID(),
        max_length=10,
        tokens=np.array([0, 1, 2, 3]),
    )

    # Can't trim more tokens than the context has.
    with pytest.raises(ValueError):
        context.skip_processing(n=999)

    # Trimming 0 tokens does nothing.
    assert (context.next_tokens == np.array([0, 1, 2, 3])).all()
    assert context.active_length == 4
    assert context.current_length == 4

    # Trimming 2 tokens should remove the first 2 tokens of prompt.
    context.skip_processing(n=2)
    assert (context.next_tokens == np.array([2, 3])).all()
    assert context.active_length == 2
    assert context.current_length == 4  # does not change

    # Can't trim prompt to 0 tokens.
    with pytest.raises(ValueError):
        context.skip_processing(n=2)


def test_context__update_beyond_chunk_size() -> None:
    # This check evaluates whether we can update this array.
    # However, behaviour with max serve for updating, is slightly
    # different than behaviour off the server, as the text context
    # moves between the api worker & server worker.
    # Before making changes to resize behaviour, ensure you
    # test with the server, not just the `generate` entrypoint.
    context = TextContext(
        request_id=RequestID(),
        max_length=10,
        tokens=np.array([0, 1, 2, 3]),
    )

    # 128, is the CHUNK_SIZE defined in context
    for i in range(128):
        context.update(i)


def test_context__reset() -> None:
    context = TextContext(
        request_id=RequestID(),
        max_length=10,
        tokens=np.array([0, 1, 2, 3]),
    )
    assert context.active_length == 4
    assert context.next_tokens.tolist() == [0, 1, 2, 3]
    context.update(4)
    assert context.active_length == 1
    assert context.next_tokens.tolist() == [4]
    context.reset()
    assert context.active_length == 5
    assert context.next_tokens.tolist() == [0, 1, 2, 3, 4]
    context.update(5)
    assert context.active_length == 1
    assert context.next_tokens.tolist() == [5]


def test_context_sampling_params_integration() -> None:
    """Tests that TextContext properly stores and maintains SamplingParams."""
    custom_params = SamplingParams(
        top_k=25,
        temperature=0.7,
        frequency_penalty=0.4,
        presence_penalty=0.2,
        repetition_penalty=1.15,
    )

    context = TextContext(
        request_id=RequestID(),
        max_length=50,
        tokens=np.array([0, 1, 2, 3, 4]),
        sampling_params=custom_params,
    )

    # Verify the sampling params persist through context operations
    context.update(5)
    assert context.sampling_params is custom_params
    assert context.sampling_params.top_k == 25

    context.reset()
    assert context.sampling_params is custom_params
    assert context.sampling_params.temperature == 0.7
    assert context.sampling_params.temperature == 0.7


def test_context_sampling_params_stop() -> None:
    """Tests that TextContext can stop on user-defined sequences."""
    custom_params = SamplingParams(stop=["This is a test"])

    context = TextContext(
        request_id=RequestID(),
        max_length=50,
        tokens=np.array([0]),
        eos_sequences=[[1, 2]],
        sampling_params=custom_params,
    )

    context.update(1)
    context.update(2)
    print(context.generated_tokens)
    assert context.is_done
    assert np.array_equal(context.generated_tokens, np.array([1, 2]))

    context = TextContext(
        request_id=RequestID(),
        max_length=50,
        tokens=np.array([0]),
        eos_sequences=[[2], [3, 1]],
        sampling_params=custom_params,
    )
    context.update(1)
    context.update(3)

    assert not context.is_done
    assert np.array_equal(context.generated_tokens, np.array([1, 3]))


def test_context_sampling_params_eos_token_ids() -> None:
    """Tests that TextContext can stop on user-defined sequences."""
    custom_params = SamplingParams(stop=["This is a test"])

    context = TextContext(
        request_id=RequestID(),
        max_length=50,
        tokens=np.array([0]),
        eos_token_ids=set([5, 4, 2]),
        sampling_params=custom_params,
    )
    context.update(1)
    context.update(2)

    assert context.is_done
    assert np.array_equal(context.generated_tokens, np.array([1, 2]))

    context = TextContext(
        request_id=RequestID(),
        max_length=50,
        tokens=np.array([0]),
        eos_token_ids=set([5, 4, 2]),
        sampling_params=custom_params,
    )
    context.update(3)
    context.update(6)

    assert not context.is_done
    assert np.array_equal(context.generated_tokens, np.array([3, 6]))


def test_sampling_params_from_input_and_generation_config_defaults_override() -> (
    None
):
    """Test that SamplingParamsGenerationConfigDefaults values override SamplingParams class defaults."""
    # Create defaults that override some SamplingParams class defaults
    generation_defaults = SamplingParamsGenerationConfigDefaults(
        temperature=0.5,
        top_k=50,
        max_new_tokens=100,
    )

    # Create SamplingParams with no user overrides
    sampling_params = SamplingParams.from_input_and_generation_config(
        SamplingParamsInput(),
        sampling_params_defaults=generation_defaults,
    )

    # Verify that generation config defaults override SamplingParams class defaults
    assert (
        sampling_params.temperature == 0.5
    )  # from generation_defaults, not 1.0 (class default)
    assert (
        sampling_params.top_k == 50
    )  # from generation_defaults, not -1 (class default)
    assert (
        sampling_params.max_new_tokens == 100
    )  # from generation_defaults, not None (class default)

    # Verify that fields not in generation_defaults retain their class defaults
    assert sampling_params.top_p == 1  # class default
    assert sampling_params.min_p == 0.0  # class default
    assert sampling_params.frequency_penalty == 0.0  # class default
    assert sampling_params.repetition_penalty == 1.0  # class default


def test_sampling_params_from_input_and_generation_config_user_override() -> (
    None
):
    """Test that user-provided values take highest priority over defaults."""
    # Create generation defaults
    generation_defaults = SamplingParamsGenerationConfigDefaults(
        temperature=0.5,
        top_k=50,
        top_p=0.9,
        max_new_tokens=100,
    )

    # Create SamplingParams with user overrides
    user_input = SamplingParamsInput(
        temperature=0.8,  # Override generation default (0.5)
        top_k=10,  # Override generation default (50)
        min_new_tokens=5,  # Not in generation defaults, overrides class default (0)
    )

    sampling_params = SamplingParams.from_input_and_generation_config(
        user_input,
        sampling_params_defaults=generation_defaults,
    )

    # Verify user values take highest priority
    assert (
        sampling_params.temperature == 0.8
    )  # user value, not generation default (0.5)
    assert (
        sampling_params.top_k == 10
    )  # user value, not generation default (50)
    assert (
        sampling_params.min_new_tokens == 5
    )  # user value, not class default (0)

    # Verify generation defaults are used when user doesn't override
    assert sampling_params.top_p == 0.9  # from generation_defaults
    assert sampling_params.max_new_tokens == 100  # from generation_defaults

    # Verify class defaults are used when neither user nor generation defaults provide values
    assert sampling_params.min_p == 0.0  # class default
    assert sampling_params.frequency_penalty == 0.0  # class default


def test_context_serializable() -> None:
    # Test that we can encode a sample TextContext with Pickle
    original_context = TextContext(
        request_id=RequestID(),
        max_length=50,
        tokens=np.array([0, 1, 2, 3, 4]),
    )

    pickle_encoded = pickle.dumps(original_context)
    pickle_decoded = pickle.loads(pickle_encoded)

    assert isinstance(pickle_decoded, TextContext)
    assert msgpack_eq(pickle_decoded, original_context)

    # Test that we can encode a sample TextContext with MsgPack
    serialize = msgpack_numpy_encoder()
    deserialize = msgpack_numpy_decoder(TextContext | TextAndVisionContext)
    msgpack_encoded = serialize(original_context)
    msgpack_decoded = deserialize(msgpack_encoded)

    assert msgpack_eq(msgpack_decoded, original_context)


def test_context_tuple_serializable() -> None:
    # Test that we can encode a tuple of (str, TextContext) with Pickle
    original_context = TextContext(
        request_id=RequestID(),
        max_length=50,
        tokens=np.array([0, 1, 2, 3, 4]),
    )
    original_tuple = ("test_key", original_context)

    pickle_encoded = pickle.dumps(original_tuple)
    pickle_decoded = pickle.loads(pickle_encoded)

    assert pickle_decoded[0] == original_tuple[0]
    assert msgpack_eq(pickle_decoded[1], original_tuple[1])

    # Test that we can encode a tuple of (str, TextContext) with MsgPack
    serialize = msgpack_numpy_encoder()
    deserialize = msgpack_numpy_decoder(
        tuple[str, TextContext | TextAndVisionContext]
    )
    msgpack_encoded = serialize(original_tuple)
    msgpack_decoded = deserialize(msgpack_encoded)

    assert msgpack_decoded[0] == original_tuple[0]
    assert msgpack_eq(msgpack_decoded[1], original_tuple[1])


def test_text_and_vision_context_serializable() -> None:
    # Test that we can encode a sample TextAndVisionContext with Pickle
    original_context = TextAndVisionContext(
        max_length=50,
        tokens=np.array([0, 0, 2, 3, 4]),
        images=[
            ImageMetadata(
                start_idx=0,
                end_idx=2,
                pixel_values=np.array([99]),
            )
        ],
        vision_token_ids=[0],
    )

    pickle_encoded = pickle.dumps(original_context)
    pickle_decoded = pickle.loads(pickle_encoded)

    assert isinstance(pickle_decoded, TextAndVisionContext)
    assert msgpack_eq(pickle_decoded, original_context)

    # Test that we can encode a sample TextAndVisionContext with MsgPack
    serialize = msgpack_numpy_encoder()
    deserialize = msgpack_numpy_decoder(TextContext | TextAndVisionContext)
    msgpack_encoded = serialize(original_context)
    msgpack_decoded = deserialize(msgpack_encoded)

    assert msgpack_eq(msgpack_decoded, original_context)


def test_text_and_vision_context_serializable_empty_pixel_values() -> None:
    # Test that we can encode a sample TextAndVisionContext with Pickle
    original_context = TextAndVisionContext(
        max_length=50,
        tokens=np.array([0, 1, 2, 3, 4]),
        images=[],
        vision_token_ids=[98],
    )

    pickle_encoded = pickle.dumps(original_context)
    pickle_decoded = pickle.loads(pickle_encoded)

    assert isinstance(pickle_decoded, TextAndVisionContext)
    assert msgpack_eq(pickle_decoded, original_context)

    # Test that we can encode a sample TextAndVisionContext with MsgPack
    serialize = msgpack_numpy_encoder()
    deserialize = msgpack_numpy_decoder(TextContext | TextAndVisionContext)
    msgpack_encoded = serialize(original_context)
    msgpack_decoded = deserialize(msgpack_encoded)

    assert msgpack_eq(msgpack_decoded, original_context)


def test_text_and_vision_context_tuple_serializable() -> None:
    # Test that we can encode a tuple of (str, TextAndVisionContext) with Pickle
    original_context = TextAndVisionContext(
        max_length=50,
        tokens=np.array([0, 0, 2, 3, 4]),
        images=[
            ImageMetadata(
                start_idx=0,
                end_idx=2,
                pixel_values=np.array([99]),
            )
        ],
        vision_token_ids=[0],
    )
    original_tuple = ("test_key", original_context)

    pickle_encoded = pickle.dumps(original_tuple)
    pickle_decoded = pickle.loads(pickle_encoded)

    assert pickle_decoded[0] == original_tuple[0]
    assert msgpack_eq(pickle_decoded[1], original_tuple[1])

    # Test that we can encode a tuple of (str, TextAndVisionContext) with MsgPack
    serialize = msgpack_numpy_encoder()
    deserialize = msgpack_numpy_decoder(
        tuple[str, TextContext | TextAndVisionContext]
    )
    msgpack_encoded = serialize(original_tuple)
    msgpack_decoded = deserialize(msgpack_encoded)

    assert msgpack_decoded[0] == original_tuple[0]
    assert msgpack_eq(msgpack_decoded[1], original_tuple[1])


def test_tts_context_msgpack_serialization_and_speech_tokens() -> None:
    """Tests that TTSContext can be serialized/deserialized with msgpack and that _speech_tokens can be written to after deserialization."""
    # Create a TTSContext with some audio prompt tokens
    audio_prompt_tokens = np.array([100, 101, 102, 103], dtype=np.int32)
    original_context = TTSContext(
        max_length=50,
        tokens=np.array([0, 1, 2, 3, 4]),
        audio_prompt_tokens=audio_prompt_tokens,
        sampling_params=SamplingParams(temperature=0.8),
    )

    # Add some initial speech tokens to the context
    initial_speech_tokens = np.array([200, 201, 202], dtype=np.int32)
    original_context.update_speech_tokens(initial_speech_tokens)

    # Verify initial state
    assert np.array_equal(
        original_context.audio_prompt_tokens, audio_prompt_tokens
    )
    assert np.array_equal(original_context.speech_tokens, initial_speech_tokens)
    assert original_context._speech_token_end_idx == 3
    assert original_context.block_counter == 1

    # Test that we can encode TTSContext with MsgPack
    serialize = msgpack_numpy_encoder()
    deserialize = msgpack_numpy_decoder(TTSContext)
    msgpack_encoded = serialize(original_context)
    msgpack_decoded = deserialize(msgpack_encoded)

    # Verify the deserialized context matches the original
    assert isinstance(msgpack_decoded, TTSContext)
    assert msgpack_eq(msgpack_decoded, original_context)
    assert np.array_equal(
        msgpack_decoded.audio_prompt_tokens, audio_prompt_tokens
    )
    assert np.array_equal(msgpack_decoded.speech_tokens, initial_speech_tokens)
    assert msgpack_decoded._speech_token_end_idx == 3
    assert msgpack_decoded.block_counter == 1

    # Test writing to the _speech_tokens array after deserialization
    new_speech_tokens = np.array([300, 301, 302, 303], dtype=np.int32)
    msgpack_decoded.update_speech_tokens(new_speech_tokens)

    # Verify that the new speech tokens were added correctly
    expected_combined_tokens = np.concatenate(
        [initial_speech_tokens, new_speech_tokens]
    )
    assert np.array_equal(
        msgpack_decoded.speech_tokens, expected_combined_tokens
    )
    assert msgpack_decoded._speech_token_end_idx == 7
    assert msgpack_decoded.block_counter == 2

    # Verify that the original context was not affected
    assert np.array_equal(original_context.speech_tokens, initial_speech_tokens)
    assert original_context._speech_token_end_idx == 3
    assert original_context.block_counter == 1


def test_vision_context_reset() -> None:
    context = TextAndVisionContext(
        max_length=50,
        tokens=np.array([0, 1, 2, 3, 4]),
        images=[
            ImageMetadata(
                start_idx=0,
                end_idx=1,
                pixel_values=np.array([10, 11, 12, 13, 14]),
            )
        ],
        vision_token_ids=[0],
    )
    assert len(context.images) == 1
    assert context.images[0].pixel_values.tolist() == [10, 11, 12, 13, 14]
    assert context.start_idx == 0
    assert context.active_length == 5
    assert context.needs_vision_encoding is True

    # The pixel values should remain set after update, but needs_vision_encoding should be False.
    context.update(5)
    assert len(context.images) == 1
    assert context.images[0].pixel_values.tolist() == [10, 11, 12, 13, 14]
    assert context.needs_vision_encoding is False
    assert context.start_idx == 5
    assert context.active_length == 1

    # The pixel values should be restored after reset.
    context.reset()
    assert len(context.images) == 1
    assert context.images[0].pixel_values.tolist() == [10, 11, 12, 13, 14]
    assert context.start_idx == 0
    assert context.active_length == 6
    assert context.needs_vision_encoding is True


def test_text_context_repr(capsys: pytest.CaptureFixture) -> None:  # type: ignore
    context = TextContext(
        request_id=RequestID(),
        tokens=np.array([0, 1, 2, 3, 4]),
        max_length=5,
    )
    print(context)
    outerr = capsys.readouterr()
    assert (
        re.match(
            r"TextContext\(request_id=.*\, start_idx=0\, active_idx=5\, end_idx=5\)",
            outerr.out,
        )
        is not None
    )


def test_context__chunked_prefill_needs_ce_edge_case() -> None:
    """Test that needs_ce behaves correctly during chunked prefill processing.

    This test reproduces the production edge case where a context in chunked prefill
    reaches the end of prompt processing (_start_idx == _prompt_len) but hasn't
    started completion generation (_completion_start_idx == _completion_end_idx),
    while status remains ACTIVE and needs_ce incorrectly returns False.
    """
    # Test parameters
    n = 32  # Initial prompt length
    chunk_size = 8  # Chunked prefill chunk size
    m = n + chunk_size + 5  # Additional tokens: m > (n + chunk_size)

    # a. Create a random prompt of length n
    initial_prompt = np.arange(n, dtype=np.int32)

    context = TextContext(
        max_length=200,  # Large enough to accommodate all tokens
        tokens=initial_prompt,
    )

    # Verify initial state
    assert context.active_length == n
    assert context.needs_ce is True
    assert context._prompt_len == n
    assert (
        context._completion_start_idx == context._completion_end_idx
    )  # No completions yet

    # b. Generate n + m tokens, where m > (n + chunk_size)
    for i in range(m):
        context.update(n + i)

    # Verify we've generated the expected number of tokens
    assert context.current_length == n + m
    assert (
        context.needs_ce is False
    )  # All original prompt processed, completion generated
    assert (
        context._completion_start_idx != context._completion_end_idx
    )  # Has completions

    # c. Reset the context object
    context.reset()

    # After reset, all tokens become the new prompt
    new_prompt_len = n + m
    assert context.active_length == new_prompt_len
    assert context._prompt_len == new_prompt_len
    assert context.needs_ce is True
    assert context.status == GenerationStatus.ACTIVE
    # Critical: completion indices are reset to equal values (the edge case setup)
    _ = context.to_generation_output()
    assert context._completion_start_idx == context._completion_end_idx

    # d. Simulate chunked prefill processing chunk by chunk
    processed_tokens = 0

    while processed_tokens < new_prompt_len:
        # Calculate current chunk size
        remaining_tokens = new_prompt_len - processed_tokens
        current_chunk_size = min(remaining_tokens, chunk_size)
        if current_chunk_size < remaining_tokens:
            context.chunk(current_chunk_size)

        # Before simulating the update call, verify needs_ce
        assert context.needs_ce is True, (
            f"needs_ce should be True when processing chunk at tokens "
            f"{processed_tokens} to {processed_tokens + current_chunk_size}"
        )

        # Simulate the chunked prefill path in update() method
        context.update(1)

        # Verify that indices were updated correctly
        assert context._start_idx == processed_tokens + current_chunk_size
        processed_tokens += current_chunk_size

        # Key test: verify needs_ce behavior after chunk processing
        if processed_tokens < new_prompt_len:
            # Still have prompt tokens to process
            assert context._active_idx == new_prompt_len
            assert context.needs_ce is True, (
                f"needs_ce should be True after processing {processed_tokens}/{new_prompt_len} tokens"
            )
            assert (
                context._completion_start_idx == context._completion_end_idx
            ), "Should have no completion tokens yet"

        else:
            # We've reached the critical edge case:
            # - All prompt tokens processed (_start_idx == _prompt_len)
            # - One completion tokens generated (_completion_start_idx == _completion_end_idx)
            # - Status is still ACTIVE
            assert context._active_idx == new_prompt_len + 1

            assert context._start_idx == context._prompt_len, (
                "Should have processed all prompt tokens"
            )
            assert context.status == GenerationStatus.ACTIVE, (
                "Status should still be ACTIVE"
            )

            assert context.needs_ce is False, (
                "Processed all prompt tokens but no completion tokens generated",
            )
            assert (
                context._completion_start_idx + 1 == context._completion_end_idx
            ), "Should have no completion tokens yet"

    # Verify final state before first completion token
    _ = context.to_generation_output()
    assert context._start_idx == new_prompt_len
    assert context.active_length == 1
    assert (
        context._completion_start_idx == context._completion_end_idx
    )  # No completions yet

    # Now simulate generating the first completion token
    # This should transition the context out of the edge case
    context.update(999)  # Generate first actual completion token

    # Verify proper transition to completion generation
    assert context.active_length == 1
    assert context.needs_ce is False  # Now we genuinely don't need CE
    assert (
        context._completion_start_idx != context._completion_end_idx
    )  # Now we have completions
    assert (
        context.status == GenerationStatus.ACTIVE
    )  # Still active, but generating completions


def test_text_and_vision_context_post_init() -> None:
    # ok (contains one <vision_token_id>)
    _ = ImageMetadata(
        start_idx=0,
        end_idx=1,
        pixel_values=np.array([99]),
    )

    # not ok since start_idx is negative
    with pytest.raises(ValueError):
        _ = ImageMetadata(
            start_idx=-1,
            end_idx=1,
            pixel_values=np.array([99]),
        )

    # not ok since there are no room for any <vision_token_id>
    with pytest.raises(ValueError):
        _ = ImageMetadata(
            start_idx=0,
            end_idx=0,
            pixel_values=np.array([99]),
        )

    # ok (no images)
    _ = TextAndVisionContext(
        max_length=50,
        tokens=np.array([0, 1, 2, 3, 4]),
        images=[],
        vision_token_ids=[98],
    )


def test_text_and_vision_context_happy_case() -> None:
    # fmt: off
    #                                      |<-- img0 --->|                         |<--- img1 -->|
    #                   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23
    tokens = np.array([51, 52, 53, 54, 97, 98, 98, 98, 98, 99, 55, 56, 57, 58, 97, 98, 98, 98, 98, 99, 59, 60, 61, 62])
    # fmt: on
    ctx = TextAndVisionContext(
        max_length=50,
        tokens=tokens,
        images=[
            ImageMetadata(
                start_idx=5,
                end_idx=9,
                pixel_values=np.array([99]),
            ),
            ImageMetadata(
                start_idx=15,
                end_idx=19,
                pixel_values=np.array([99]),
            ),
        ],
        vision_token_ids=[98],
    )

    assert ctx.compute_image_aligned_idx(7) == 5
    assert ctx.compute_image_aligned_idx(8) == 5
    assert ctx.compute_image_aligned_idx(9) == 9
    assert ctx.compute_image_aligned_idx(10) == 10

    assert ctx.compute_image_aligned_idx(13) == 13
    assert ctx.compute_image_aligned_idx(14) == 14
    assert ctx.compute_image_aligned_idx(15) == 15
    assert ctx.compute_image_aligned_idx(17) == 15
    assert ctx.compute_image_aligned_idx(18) == 15
    assert ctx.compute_image_aligned_idx(19) == 19
    assert ctx.compute_image_aligned_idx(20) == 20

    assert ctx.image_idx == 0
    assert ctx.needs_vision_encoding is True
    assert len(ctx.next_images) == 2

    ctx.skip_processing(9)
    assert ctx.image_idx == 1
    assert ctx.needs_vision_encoding is True
    assert len(ctx.next_images) == 1

    ctx.skip_processing(5)
    assert ctx.image_idx == 1
    assert ctx.needs_vision_encoding is True
    assert len(ctx.next_images) == 1

    ctx.skip_processing(5)
    assert ctx.image_idx == 2
    assert ctx.needs_vision_encoding is False
    assert len(ctx.next_images) == 0


def test_text_and_vision_context_sad_case() -> None:
    # fmt: off
    #                                      |<-- img0 --->|                         |<--- img1 -->|
    #                   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23
    tokens = np.array([51, 52, 53, 54, 97, 98, 98, 98, 98, 99, 55, 56, 57, 58, 97, 98, 98, 98, 98, 99, 59, 60, 61, 62])
    # fmt: on

    with pytest.raises(ValueError, match="Images must be non-overlapping"):
        _ = TextAndVisionContext(
            max_length=50,
            tokens=tokens,
            images=[
                ImageMetadata(
                    start_idx=5,
                    end_idx=9,
                    pixel_values=np.array([99]),
                ),
                # This overlaps with img0
                ImageMetadata(
                    start_idx=9,
                    end_idx=19,
                    pixel_values=np.array([99]),
                ),
            ],
            vision_token_ids=[98],
        )

    with pytest.raises(ValueError, match="Images must be sorted"):
        _ = TextAndVisionContext(
            max_length=50,
            tokens=tokens,
            images=[
                ImageMetadata(
                    start_idx=15,
                    end_idx=19,
                    pixel_values=np.array([99]),
                ),
                ImageMetadata(
                    start_idx=5,
                    end_idx=9,
                    pixel_values=np.array([99]),
                ),
            ],
            vision_token_ids=[98],
        )

    with pytest.raises(
        ValueError, match="Images must be before the end of the token array"
    ):
        _ = TextAndVisionContext(
            max_length=50,
            tokens=tokens,
            images=[
                ImageMetadata(
                    start_idx=20,
                    end_idx=25,
                    pixel_values=np.array([99]),
                ),
            ],
            vision_token_ids=[98],
        )

    ctx = TextAndVisionContext(
        max_length=50,
        tokens=tokens,
        images=[
            ImageMetadata(
                start_idx=5,
                end_idx=9,
                pixel_values=np.array([99]),
            ),
            ImageMetadata(
                start_idx=15,
                end_idx=19,
                pixel_values=np.array([99]),
            ),
        ],
        vision_token_ids=[98],
    )
    with pytest.raises(
        ValueError,
        match="It is invalid for the active_idx \(7\) to bisect an image \(ImageMetadata\(start_idx=5, end_idx=9",
    ):
        ctx.chunk(7)

    with pytest.raises(
        ValueError,
        match="Images must be filled with <vision_token_id>",
    ):
        _ = TextAndVisionContext(
            max_length=50,
            tokens=tokens,
            images=[
                ImageMetadata(
                    start_idx=5, end_idx=9, pixel_values=np.array([99])
                ),
            ],
            vision_token_ids=[123],
        )


def does_not_raise_due_to_check_in_property_method() -> None:
    """This test ensures that `isinstance`, `hasattr`, and other class introspection
    methods do not execute the body of `get_last_generated_token()` and throw an
    exception. If `get_last_generated_token()` was instead a @property method
    called via `ctx.last_generated_token`, this test would fail.
    """

    ctx = TTSContext(
        max_length=10,
        tokens=np.array([0, 1, 2, 3]),
    )

    # Verify that calling get_last_generated_token() raises an exception
    with pytest.raises(ValueError, match="No tokens have been generated"):
        _ = ctx.get_last_generated_token()

    # Try very hard to execute body of last_generated_token to throw exception
    _ = hasattr(ctx, "get_last_generated_token")
    _ = getattr(ctx, "get_last_generated_token", None)
    # Protocol structural checks should NOT trigger the method body!
    # (TextGenerationContext and VLMTextGenerationContext are Protocols)
    _ = isinstance(ctx, TextGenerationContext)
    # The original bug report indicated that MAX threw a ValueError in call to
    # isinstance(ctx, VLMTextGenerationContext) so we are validating this case here.
    # See GENAI-318 for details.
    _ = isinstance(ctx, VLMTextGenerationContext)
