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
"""LFM3 speech output: the depth decoder and the codec decoder.

A speech frame is one code per codebook. The depth decoder reads the
language model's hidden state and scores the codebooks still hidden from
it; MaskGIT sampling reveals them in a random order over a few passes.
The LiquidCodec decoder turns frames into a spectrogram, which the host
inverts into 24 kHz samples.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from itertools import pairwise

import numpy as np
import numpy.typing as npt
from max import functional as F
from max.dtype import DType
from max.graph.weights import WeightData, Weights
from max.nn import Embedding, Linear, Module, ModuleList
from max.nn.norm import RMSNorm
from max.nn.norm.rms_norm import rms_norm
from max.tensor import Tensor

from .lfm3_audio import Lfm3AudioEncoderStack
from .model_config import Lfm3AudioConfig, Lfm3CodecConfig, Lfm3SpeechConfig

_LFM3_TO_STACK = (
    ("self_attn.q_norm.", "self_attn.q_layernorm."),
    ("self_attn.k_norm.", "self_attn.k_layernorm."),
    ("self_attn.o_proj.", "self_attn.out_proj."),
    ("conv.conv.weight", "conv.conv"),
)


def stack_weight_name(name: str) -> str:
    """Renames an LFM3 layer's weight to LFM2 naming.

    The names are the ones :class:`Lfm3AudioEncoderStack` uses.
    """
    for old, new in _LFM3_TO_STACK:
        name = name.replace(old, new)
    return name


def depth_decoder_weights(weights: Weights) -> dict[str, WeightData]:
    """Returns the depth decoder's weights from a checkpoint.

    Reads an LFM3-Audio checkpoint, which keeps them at the top level, or an
    LFM3-Omni one.
    """
    out: dict[str, WeightData] = {}
    for name, weight in weights.items():
        if name.startswith("model.depth_decoder."):
            name = name.removeprefix("model.depth_decoder.")
        elif name.startswith("depthformer."):
            name = name.removeprefix("depthformer.")
        elif not name.startswith(
            ("depth_linear.", "mask_embedding", "depth_embeddings.")
        ):
            continue
        # LFM3-Audio's untied heads, which the reference doesn't read.
        if not name.endswith("to_logits.weight"):
            out[stack_weight_name(name)] = weight.data()
    return out


class Lfm3CodebookEmbedding(Module[[Tensor], Tensor]):
    """One codebook's embedding, and its head tied to it."""

    def __init__(self, size: int, width: int, eps: float) -> None:
        self.embedding = Embedding(size, dim=width)
        self.embedding_norm = RMSNorm(width, eps=eps)

    def forward(self, codes: Tensor) -> Tensor:
        """Embeds frame codes."""
        return self.embedding(codes)

    def logits(self, h: Tensor) -> Tensor:
        """Returns the head's float32 logits over the codebook."""
        norm = self.embedding_norm
        # The head's norm scales in float32 and rounds once.
        normed = rms_norm(h, norm.weight, norm.eps, multiply_before_cast=True)
        return F.cast(normed @ self.embedding.weight.T, DType.float32)


class Lfm3DepthDecoder(Module[..., Tensor]):
    """Scores every codebook of a batch of frames.

    Each codebook gets its slot of the projected hidden state plus its
    known code's embedding, or the mask embedding while hidden. A
    bidirectional LFM3 stack over each frame's slots feeds the codebooks'
    heads.
    """

    def __init__(self, config: Lfm3AudioConfig) -> None:
        speech = config.speech
        width = speech.depth.hidden_size
        self.num_codebooks = speech.num_codebooks
        self.depth_linear = Linear(
            config.hidden_size, self.num_codebooks * width, bias=True
        )
        self.mask_embedding = Tensor.zeros([width])
        self.model = Lfm3AudioEncoderStack(speech.depth)
        self.depth_embeddings = ModuleList(
            [
                Lfm3CodebookEmbedding(
                    speech.codebook_size, width, speech.depth.norm_eps
                )
                for _ in range(self.num_codebooks)
            ]
        )

    def forward(
        self,
        hidden: Tensor,
        rows: Tensor,
        codes: Tensor,
        masked: Tensor,
        position_ids: Tensor,
        band_taps: Tensor,
    ) -> Tensor:
        """Returns float32 logits for the frames at ``rows`` of ``hidden``.

        The logits are ``[frames, codebooks, codebook_size]``.
        """
        c = self.num_codebooks
        width = int(self.mask_embedding.shape[0])
        h = self.depth_linear(F.gather(hidden, rows, axis=0))
        embedded = F.stack(
            [e(codes[:, i]) for i, e in enumerate(self.depth_embeddings)],
            axis=1,
        )
        slots = h.reshape((-1, c, width)) + F.where(
            masked.reshape((-1, c, 1)), self.mask_embedding, embedded
        )
        frames = codes.shape[0]
        slots = F.rebind(
            slots.reshape((-1, width)), (position_ids.shape[0], width)
        )
        out = self.model(slots, position_ids, band_taps)
        out = F.rebind(out, (frames * c, width)).reshape((frames, c, width))
        return F.stack(
            [e.logits(out[:, i]) for i, e in enumerate(self.depth_embeddings)],
            axis=1,
        )


class _Quantizer(Module[[Tensor], Tensor]):
    def __init__(self, config: Lfm3CodecConfig) -> None:
        latents = len(config.fsq_levels) * config.num_codebooks
        self.project_out = Linear(latents, config.quantizer_dim, bias=True)

    def forward(self, latents: Tensor) -> Tensor:
        return self.project_out(latents)


class _CodecHead(Module[..., Tensor]):
    def __init__(self, config: Lfm3CodecConfig) -> None:
        self.lfm = Lfm3AudioEncoderStack(config.decoder)
        self.lin = Linear(
            config.decoder.hidden_size, config.n_fft + 2, bias=True
        )

    def forward(
        self, h: Tensor, position_ids: Tensor, band_taps: Tensor
    ) -> Tensor:
        return self.lin(self.lfm(h, position_ids, band_taps))


class Lfm3CodecDecoder(Module[..., Tensor]):
    """The LiquidCodec decoder up to its spectrogram.

    Each frame's latents project up and repeat over its spectrogram frames,
    which a causal LFM3 stack turns into log magnitudes and phases.
    """

    def __init__(self, config: Lfm3CodecConfig) -> None:
        self.frame_stack = config.frame_stack
        self.quantizer = _Quantizer(config)
        self.from_quantizer = Linear(
            config.quantizer_dim, config.decoder.hidden_size, bias=True
        )
        self.decoder = _CodecHead(config)

    def forward(
        self, latents: Tensor, position_ids: Tensor, band_taps: Tensor
    ) -> Tensor:
        """Returns the spectrogram rows of ``[frames, latents]`` latents.

        The rows are ``[spectrogram frames, n_fft + 2]`` float32: the log
        magnitudes, then the phases.
        """
        h = self.from_quantizer(
            self.quantizer(F.cast(latents, self.from_quantizer.weight.dtype))
        )
        frames, width = h.shape
        h = F.broadcast_to(
            h.reshape((frames, 1, width)), (frames, self.frame_stack, width)
        ).reshape((-1, width))
        h = F.rebind(h, (position_ids.shape[0], width))
        return F.cast(self.decoder(h, position_ids, band_taps), DType.float32)


def codec_decoder_weights(weights: Weights) -> dict[str, WeightData]:
    """Returns the codec decoder's weights from the codec checkpoint."""
    return {
        stack_weight_name(name): weight.data()
        for name, weight in weights.items()
        if name.startswith(
            (
                "quantizer.project_out.",
                "from_quantizer.",
                "decoder.lfm.",
                "decoder.lin.",
            )
        )
    }


def fsq_latents(
    codes: npt.NDArray[np.int64], levels: Sequence[int]
) -> npt.NDArray[np.float32]:
    """Returns the finite scalar quantizer's latents of the codes.

    Maps ``[frames, codebooks]`` codes to ``[frames, codebooks * len(levels)]``
    latents.
    """
    levels_ = np.asarray(levels)
    basis = np.cumprod([1, *levels[:-1]])
    half = levels_ // 2
    digits = (codes[..., None] // basis) % levels_
    return ((digits - half) / half).reshape(len(codes), -1).astype(np.float32)


def hann_window(n: int) -> npt.NDArray[np.float32]:
    """Returns the periodic Hann window, as ``torch.hann_window`` does."""
    return (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)).astype(np.float32)


def istft(
    spectrogram: npt.NDArray[np.float32], hop: int
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Returns the overlap-added samples and their squared-window envelope.

    The input is ``[frames, n_fft + 2]`` rows of log magnitudes then phases.
    """
    log_magnitude, phase = np.split(spectrogram, 2, axis=-1)
    spectra = np.exp(log_magnitude + 1j * phase)
    n_fft = 2 * (spectra.shape[-1] - 1)
    window = hann_window(n_fft)
    frames = np.fft.irfft(spectra, axis=-1).astype(np.float32) * window
    count = len(frames)
    audio = np.zeros((count - 1) * hop + n_fft, dtype=np.float32)
    envelope = np.zeros_like(audio)
    for start in range(0, n_fft, hop):
        span = slice(start, start + count * hop)
        audio[span] += frames[:, start : start + hop].reshape(-1)
        envelope[span] += np.tile(window[start : start + hop] ** 2, count)
    return audio, envelope


class CodecStream:
    """One request's speech, decoded a few frames at a time.

    The decoder's receptive field spans ``context`` frames, so redecoding
    them ahead of the new frames reproduces decoding the whole stream; the
    samples later frames still overlap stay held back.
    """

    def __init__(self, config: Lfm3CodecConfig) -> None:
        self.config = config
        self.frames: list[npt.NDArray[np.int64]] = []
        self.decoded = 0
        self._tail: tuple[npt.NDArray[np.float32], ...] = ()
        self._started = False

    @property
    def pending(self) -> int:
        """The number of frames waiting for the codec."""
        return len(self.frames) - self.decoded

    def window(self) -> tuple[npt.NDArray[np.int64], int]:
        """Returns the codes to decode next, with their context.

        Also returns the first code's index in the stream.
        """
        start = max(0, self.decoded - self.config.context_frames)
        return np.stack(self.frames[start:]), start

    def emit(
        self, spectrogram: npt.NDArray[np.float32], finalize: bool
    ) -> npt.NDArray[np.float32]:
        """Returns the samples the new frames' spectrogram rows finish."""
        hop = self.config.hop_length
        trim = (self.config.n_fft - hop) // 2
        self.decoded = len(self.frames)
        if len(spectrogram):
            audio, envelope = istft(spectrogram, hop)
        elif self._tail:
            audio = np.zeros(2 * trim, dtype=np.float32)
            envelope = np.zeros_like(audio)
        else:
            return np.zeros(0, dtype=np.float32)
        if self._tail:
            held, held_envelope = self._tail
            audio[: len(held)] += held
            envelope[: len(held)] += held_envelope
        if not self._started:
            audio, envelope = audio[trim:], envelope[trim:]
            self._started = True
        if finalize:
            audio, envelope = audio[:-trim], envelope[:-trim]
            self._tail = ()
        else:
            self._tail = (audio[-2 * trim :], envelope[-2 * trim :])
            audio, envelope = audio[: -2 * trim], envelope[: -2 * trim]
        return audio / envelope


def depth_layout(
    frames: int, codebooks: int
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """Returns the depth decoder's position ids and attention band.

    Each frame's slots are positions ``0..codebooks-1`` and see each other.
    """
    positions = np.tile(np.arange(codebooks, dtype=np.int32), frames)
    band = np.arange(frames, dtype=np.int32)[:, None] * codebooks + np.arange(
        codebooks, dtype=np.int32
    )
    return positions, np.repeat(band, codebooks, axis=0)


def sample_codes(
    logits: npt.NDArray[np.float32],
    histories: Sequence[npt.NDArray[np.int64]],
    rngs: Sequence[np.random.Generator],
    config: Lfm3SpeechConfig,
) -> npt.NDArray[np.int64]:
    """Samples one code per row from ``[rows, codebook_size]`` logits.

    A code in its row's history is penalized as Hugging Face's
    ``RepetitionPenaltyLogitsProcessor`` does; then the logits are tempered
    and cut to the top k. No temperature and no top k is greedy.
    """
    scores = logits.astype(np.float32, copy=True)
    penalty = config.repetition_penalty
    if penalty != 1.0:
        for row, history in enumerate(histories):
            if len(history):
                seen = np.unique(history)
                s = scores[row, seen]
                scores[row, seen] = np.where(s < 0, s * penalty, s / penalty)
    temperature, top_k = config.temperature, config.top_k
    if temperature == 0 or (temperature is None and not top_k):
        return scores.argmax(-1)
    if temperature:
        scores /= temperature
    if top_k:
        kth = np.partition(scores, -top_k, axis=-1)[:, -top_k, None]
        scores = np.where(scores < kth, -np.inf, scores)
    probs = np.exp(scores - scores.max(-1, keepdims=True))
    cdf = np.cumsum(probs, axis=-1)
    draws = np.array([rng.random() for rng in rngs]) * cdf[:, -1]
    return np.minimum(
        (cdf < draws[:, None]).sum(-1), scores.shape[-1] - 1
    ).astype(np.int64)


def maskgit_frames(
    score: Callable[
        [npt.NDArray[np.int64], npt.NDArray[np.bool_]], npt.NDArray[np.float32]
    ],
    histories: Sequence[npt.NDArray[np.int64]],
    rngs: Sequence[np.random.Generator],
    config: Lfm3SpeechConfig,
) -> npt.NDArray[np.int64]:
    """Samples one frame per row with MaskGIT.

    Each row reveals its codebooks in its own random order, as many at a
    time as ``config.maskgit_schedule`` says; each pass scores the hidden
    codebooks given the codes revealed so far.

    Args:
        score: Maps the codes so far and the hidden mask, each ``[rows,
            codebooks]``, to the depth decoder's ``[rows, codebooks,
            codebook_size]`` logits.
        histories: Each row's previous frames, ``[frames, codebooks]``.
        rngs: Each row's random generator.
        config: The speech config.
    """
    rows, c = len(rngs), config.num_codebooks
    codes = np.zeros((rows, c), dtype=np.int64)
    revealed = np.zeros((rows, c), dtype=bool)
    order = np.stack([rng.permutation(c) for rng in rngs])
    bounds = np.cumsum([0, *config.maskgit_schedule])
    for start, end in pairwise(bounds):
        logits = score(codes, ~revealed)
        for j in range(start, end):
            picked = order[:, j]
            codes[np.arange(rows), picked] = sample_codes(
                logits[np.arange(rows), picked],
                [h[:, k] for h, k in zip(histories, picked, strict=True)],
                rngs,
                config,
            )
        revealed[np.arange(rows)[:, None], order[:, start:end]] = True
    return codes


class SpeechMode(str, Enum):
    """What a reply's next slot holds: text, speech, or blocks of each."""

    TEXT = "text"
    AUDIO = "audio"
    INTERLEAVED = "interleaved"


@dataclass
class SpeechState:
    """One request's place in a spoken reply.

    Tracks what the reference's generate loop tracks: the mode, the interleaved
    block's remaining slots, the frames spoken so far and the codec stream they
    feed.
    """

    config: Lfm3SpeechConfig
    mode: SpeechMode
    rng: np.random.Generator
    stream: CodecStream
    history: deque[npt.NDArray[np.int64]]
    slots_left: int
    speaking: bool = False
    text_done: bool = False
    fed: dict[int, npt.NDArray[np.int64]] = field(default_factory=dict)
    """The codes of each frame, by the position of its placeholder."""

    @classmethod
    def start(
        cls,
        last_prompt_token: int,
        rng: np.random.Generator,
        speech: Lfm3SpeechConfig,
        codec: Lfm3CodecConfig,
    ) -> SpeechState:
        """Starts in the mode the prompt's last token opens."""
        if last_prompt_token in (
            speech.audio_start_token_id,
            speech.audio_out_token_id,
        ):
            mode = SpeechMode.AUDIO
        elif last_prompt_token == speech.mixed_start_token_id:
            mode = SpeechMode.INTERLEAVED
        else:
            mode = SpeechMode.TEXT
        return cls(
            config=speech,
            mode=mode,
            rng=rng,
            stream=CodecStream(codec),
            history=deque(maxlen=speech.repetition_window),
            slots_left=speech.interleaved_n_text,
        )

    @property
    def speaks(self) -> bool:
        """Whether the next slot is a speech frame."""
        return self.mode is SpeechMode.AUDIO or (
            self.mode is SpeechMode.INTERLEAVED and self.speaking
        )

    def past_frames(self) -> npt.NDArray[np.int64]:
        """Returns the frames the repetition penalty covers.

        The frames are ``[frames, codebooks]``.
        """
        if not self.history:
            return np.zeros((0, self.config.num_codebooks), dtype=np.int64)
        return np.stack(self.history)

    def advance(
        self, position: int, token: int, frame: npt.NDArray[np.int64] | None
    ) -> int:
        """Takes the slot at ``position`` and returns the token that holds it.

        The slot holds ``frame`` when it is speech, else the sampled ``token``.

        An end-of-audio code ends the speech: the slot holds an all
        end-of-audio frame, which the codec never sees. Outside an
        interleaved block, ``<|text_end|>`` ends the reply, so the slot
        holds ``<|im_end|>`` instead.
        """
        speech = self.config
        interleaving = self.mode is SpeechMode.INTERLEAVED
        if frame is not None:
            if (frame == speech.audio_eos_code).any():
                frame = np.full_like(frame, speech.audio_eos_code)
                self.mode = SpeechMode.TEXT
            else:
                self.history.append(frame)
                self.stream.frames.append(frame)
            self.fed[position] = frame
            token = speech.audio_out_token_id
        elif token == speech.text_end_token_id:
            if interleaving:
                self.text_done = self.speaking = True
            else:
                token = speech.end_token_id
        if interleaving and not self.text_done:
            self.slots_left -= 1
            if self.slots_left == 0:
                self.speaking = not self.speaking
                self.slots_left = (
                    speech.interleaved_n_audio
                    if self.speaking
                    else speech.interleaved_n_text
                )
        return token
