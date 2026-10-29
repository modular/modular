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
"""Tests LFM3's host-side speech frame sampling and codec streaming."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from max.pipelines.architectures.lfm3_audio.model_config import (
    Lfm3CodecConfig,
    Lfm3SpeechConfig,
    Lfm3StackConfig,
)
from max.pipelines.architectures.lfm3_audio.speech import (
    CodecStream,
    SpeechMode,
    SpeechState,
    depth_layout,
    fsq_latents,
    maskgit_frames,
    sample_codes,
)


def _speech(
    temperature: float | None = 0.0,
    top_k: int | None = None,
    repetition_penalty: float = 1.0,
) -> Lfm3SpeechConfig:
    return Lfm3SpeechConfig(
        depth=Lfm3StackConfig(
            hidden_size=8,
            intermediate_size=16,
            layer_types=["full_attention"],
            num_attention_heads=2,
            num_key_value_heads=1,
            norm_eps=1e-5,
            rope_theta=1e6,
            conv_kernel_size=3,
            sliding_window=12,
        ),
        num_codebooks=12,
        codebook_size=6,
        audio_eos_code=5,
        maskgit_schedule=[4, 4, 4],
        interleaved_n_text=6,
        interleaved_n_audio=12,
        audio_out_token_id=1,
        audio_start_token_id=2,
        mixed_start_token_id=3,
        text_end_token_id=4,
        end_token_id=9,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        repetition_window=64,
    )


def _empty_history(rows: int) -> list[npt.NDArray[np.int64]]:
    return [np.zeros(0, dtype=np.int64)] * rows


def _rngs(rows: int) -> list[np.random.Generator]:
    return [np.random.default_rng(row) for row in range(rows)]


def test_greedy_takes_the_argmax() -> None:
    logits = np.array([[0.1, 2.0, 0.5], [3.0, -1.0, 0.0]], dtype=np.float32)
    codes = sample_codes(logits, _empty_history(2), _rngs(2), _speech())
    assert codes.tolist() == [1, 0]


def test_repetition_penalty_demotes_seen_codes() -> None:
    logits = np.array([[2.0, 1.5, -1.0]], dtype=np.float32)
    history = [np.array([0, 0], dtype=np.int64)]
    codes = sample_codes(
        logits, history, _rngs(1), _speech(repetition_penalty=2.0)
    )
    assert codes.tolist() == [1]


def test_repetition_penalty_scales_negative_logits_up() -> None:
    logits = np.array([[-1.0, -1.5]], dtype=np.float32)
    history = [np.array([0], dtype=np.int64)]
    codes = sample_codes(
        logits, history, _rngs(1), _speech(repetition_penalty=2.0)
    )
    assert codes.tolist() == [1]


def test_top_1_sampling_is_greedy() -> None:
    logits = np.random.default_rng(0).normal(size=(64, 6)).astype(np.float32)
    codes = sample_codes(
        logits, _empty_history(64), _rngs(64), _speech(1.0, top_k=1)
    )
    np.testing.assert_array_equal(codes, logits.argmax(-1))


def test_sampling_stays_in_the_top_k() -> None:
    logits = np.tile(np.arange(6, dtype=np.float32), (256, 1))
    codes = sample_codes(
        logits, _empty_history(256), _rngs(256), _speech(1.0, top_k=2)
    )
    assert set(codes.tolist()) == {4, 5}


def test_sampling_follows_the_row_generator() -> None:
    logits = np.zeros((1, 6), dtype=np.float32)
    first = sample_codes(logits, _empty_history(1), _rngs(1), _speech(1.0))
    again = sample_codes(logits, _empty_history(1), _rngs(1), _speech(1.0))
    assert first.tolist() == again.tolist()


def test_maskgit_reveals_codebooks_on_schedule() -> None:
    config = _speech()
    rows = 3
    preferred = np.random.default_rng(1).integers(0, 6, size=(rows, 12))
    hidden_counts: list[list[int]] = []

    def score(
        codes: npt.NDArray[np.int64], masked: npt.NDArray[np.bool_]
    ) -> npt.NDArray[np.float32]:
        hidden_counts.append(masked.sum(-1).tolist())
        assert (codes[masked] == 0).all()
        return np.eye(6, dtype=np.float32)[preferred]

    history = np.zeros((0, 12), dtype=np.int64)
    codes = maskgit_frames(score, [history] * rows, _rngs(rows), config)
    assert hidden_counts == [[12] * rows, [8] * rows, [4] * rows]
    np.testing.assert_array_equal(codes, preferred)


def test_depth_layout_bands_each_frame() -> None:
    positions, band = depth_layout(2, 3)
    assert positions.tolist() == [0, 1, 2, 0, 1, 2]
    assert band.tolist() == [[0, 1, 2]] * 3 + [[3, 4, 5]] * 3


def test_fsq_latents_spell_codes_in_mixed_radix() -> None:
    codes = np.array([[0, 8 * 7 * 6 * 6 - 1], [1 + 8 * 3, 7]], dtype=np.int64)
    latents = fsq_latents(codes, [8, 7, 6, 6])
    np.testing.assert_allclose(
        latents,
        [
            [-1, -1, -1, -1, 3 / 4, 1, 2 / 3, 2 / 3],
            [-3 / 4, 0, -1, -1, 3 / 4, -1, -1, -1],
        ],
    )


def _codec() -> Lfm3CodecConfig:
    return Lfm3CodecConfig(
        decoder=_speech().depth,
        num_codebooks=12,
        fsq_levels=[8, 7, 6, 6],
        quantizer_dim=256,
        n_fft=64,
        hop_length=16,
        frame_stack=2,
        sample_rate=24000,
    )


def _state(last_prompt_token: int) -> SpeechState:
    return SpeechState.start(
        last_prompt_token, np.random.default_rng(0), _speech(), _codec()
    )


def test_speech_starts_in_the_mode_its_marker_opens() -> None:
    assert _state(3).mode is SpeechMode.INTERLEAVED
    assert _state(2).mode is SpeechMode.AUDIO
    assert _state(7).mode is SpeechMode.TEXT


def test_interleaved_reply_alternates_blocks_until_text_ends() -> None:
    state = _state(3)
    frame = np.ones(12, dtype=np.int64)
    kinds = []
    for position in range(40):
        speaks = state.speaks
        kinds.append("A" if speaks else "T")
        token = state.advance(
            position, 4 if position == 20 else 7, frame if speaks else None
        )
        assert token == (1 if speaks else 4 if position == 20 else 7)
    assert "".join(kinds) == "T" * 6 + "A" * 12 + "T" * 3 + "A" * 19
    assert len(state.stream.frames) == 31
    assert sorted(state.fed) == [p for p, k in enumerate(kinds) if k == "A"]


def test_end_of_audio_frame_ends_speech_unheard() -> None:
    state = _state(2)
    frame = np.zeros(12, dtype=np.int64)
    frame[3] = 5
    assert state.advance(10, 0, frame) == 1
    assert state.mode is SpeechMode.TEXT
    assert not state.stream.frames
    assert (state.fed[10] == 5).all()


def test_text_end_outside_a_block_ends_the_reply() -> None:
    assert _state(7).advance(0, 4, None) == 9


def test_codec_stream_matches_one_shot_decode() -> None:
    config = _codec()
    spectrogram = (
        np.random.default_rng(0).normal(size=(40, 66)).astype(np.float32) * 0.1
    )
    whole = CodecStream(config).emit(spectrogram, finalize=True)
    stream = CodecStream(config)
    pieces = [
        stream.emit(spectrogram[start : start + 6], finalize=start == 36)
        for start in range(0, 40, 6)
    ]
    np.testing.assert_allclose(np.concatenate(pieces), whole, atol=1e-5)
    assert len(whole) == 40 * 16
    assert not len(stream.emit(spectrogram[:0], finalize=True))
