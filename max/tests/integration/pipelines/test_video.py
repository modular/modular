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

"""Tests for the shared video decoding used by every video pipeline.

This is the single implementation of container opening, frame subsampling,
and bounded decoding, so the memory-safety and robustness properties every
model relies on are pinned here rather than re-tested per model.
"""

from __future__ import annotations

import base64
import io
import os
import tempfile
from fractions import Fraction
from pathlib import Path
from unittest.mock import MagicMock, patch

import av
import numpy as np
import numpy.typing as npt
import pytest
from max.pipelines.context import (
    InputError,
    MediaBudgetExceeded,
    MediaDecodeError,
    decode_video_frames,
    open_video_container,
    plan_frame_indices,
    probe_video,
    validate_video,
)
from max.pipelines.context import video as video_module

WIDTH = 32
HEIGHT = 32
FRAME_BYTES = WIDTH * HEIGHT * 3


def _shift_and_mux(
    container: av.container.OutputContainer,
    packet: av.Packet[av.VideoStream],
    offset: int,
) -> None:
    if offset and packet.pts is not None:
        packet.pts += offset
        if packet.dts is not None:
            packet.dts += offset
    container.mux(packet)


def _make_video_bytes(
    width: int = WIDTH,
    height: int = HEIGHT,
    num_frames: int = 8,
    rate: int = 24,
    container_format: str = "mp4",
    start_offset: int = 0,
) -> bytes:
    """Encodes a synthetic H.264 clip whose frames brighten monotonically.

    ``start_offset`` shifts every packet timestamp, in stream time-base units,
    to model containers (MPEG-TS) whose timeline does not start at zero.
    """
    buf = io.BytesIO()
    container = av.open(buf, mode="w", format=container_format)
    stream = container.add_stream("libx264", rate=rate)
    assert isinstance(stream, av.VideoStream)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    for i in range(num_frames):
        fill = min(i * (240 // max(num_frames - 1, 1)), 255)
        arr = np.full((height, width, 3), fill_value=fill, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for packet in stream.encode(frame):
            _shift_and_mux(container, packet, start_offset)
    for packet in stream.encode():
        _shift_and_mux(container, packet, start_offset)
    container.close()
    return buf.getvalue()


def _make_container(codec_names: list[str]) -> MagicMock:
    container = MagicMock(spec=av.container.InputContainer)
    video_streams = []
    for codec_name in codec_names:
        stream = MagicMock()
        stream.codec_context.name = codec_name
        video_streams.append(stream)
    container.streams.video = video_streams
    return container


class TestOpenVideoContainer:
    def test_allowed_codec_passes(self) -> None:
        container = _make_container(["h264"])
        with patch(
            "max.pipelines.context.video.av.open", return_value=container
        ):
            result = open_video_container("fake_path")
        assert result is container
        container.close.assert_not_called()

    def test_blocked_codec_raises(self) -> None:
        container = _make_container(["magicyuv"])
        with patch(
            "max.pipelines.context.video.av.open", return_value=container
        ):
            with pytest.raises(InputError, match="magicyuv"):
                open_video_container("fake_path")
        container.close.assert_called_once()

    def test_opens_in_read_mode(self) -> None:
        container = _make_container(["h264"])
        with patch(
            "max.pipelines.context.video.av.open", return_value=container
        ) as mock_open:
            open_video_container("fake_path")
        mock_open.assert_called_once_with("fake_path", mode="r")


class TestProbeVideo:
    def test_reads_header(self) -> None:
        info = probe_video(
            _make_video_bytes(width=64, height=48, num_frames=8, rate=24)
        )
        assert info.fps == 24.0
        assert info.num_frames == 8
        assert info.width == 64
        assert info.height == 48
        assert info.frame_bytes == 64 * 48 * 3

    def test_frame_count_falls_back_to_duration(self) -> None:
        """MPEG-TS declares no frame count; duration still yields an estimate."""
        video_bytes = _make_video_bytes(
            num_frames=24, container_format="mpegts"
        )
        # Premise: the stream carries no frame count. If PyAV starts reporting
        # one, this no longer exercises the duration fallback -- update it.
        with av.open(io.BytesIO(video_bytes)) as container:
            assert container.streams.video[0].frames == 0

        # The estimate need not be exact -- the decode corrects it -- but it
        # must be a usable count rather than the "unknown" sentinel.
        assert probe_video(video_bytes).num_frames > 0

    def test_garbage_raises_input_error(self) -> None:
        with pytest.raises(InputError, match="invalid or unreadable video"):
            probe_video(b"definitely not a video")

    def test_accepts_data_uri(self) -> None:
        raw = _make_video_bytes(num_frames=8)
        uri = "data:video/mp4;base64," + base64.b64encode(raw).decode()
        assert probe_video(uri).num_frames == 8


class TestValidateVideo:
    def test_valid_video_passes(self) -> None:
        validate_video(_make_video_bytes(num_frames=4))

    def test_garbage_raises(self) -> None:
        with pytest.raises(InputError, match="invalid or unreadable video"):
            validate_video(b"not a video at all")

    def test_empty_bytes_raises(self) -> None:
        with pytest.raises(InputError, match="invalid or unreadable video"):
            validate_video(b"")

    def test_validation_decodes_nothing(self) -> None:
        with patch.object(av.container.InputContainer, "decode") as decode:
            validate_video(_make_video_bytes(num_frames=4))
        decode.assert_not_called()

    def test_frame_larger_than_budget_is_rejected_before_decoding(self) -> None:
        video_bytes = _make_video_bytes(num_frames=4)
        with (
            patch.object(av.container.InputContainer, "decode") as decode,
            pytest.raises(MediaBudgetExceeded, match="more than the maximum"),
        ):
            validate_video(video_bytes, max_decoded_bytes=FRAME_BYTES - 1)
        decode.assert_not_called()

    def test_frame_within_budget_passes(self) -> None:
        validate_video(
            _make_video_bytes(num_frames=4), max_decoded_bytes=FRAME_BYTES
        )

    def test_unreported_dimensions_fail_closed_under_a_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            video_module,
            "probe_video",
            lambda source: video_module.VideoStreamInfo(
                fps=24.0, num_frames=4, width=0, height=0
            ),
        )
        with pytest.raises(MediaDecodeError, match="frame dimensions"):
            validate_video(
                _make_video_bytes(num_frames=4), max_decoded_bytes=FRAME_BYTES
            )

    def test_server_faults_are_not_reported_as_bad_input(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def out_of_memory(*args: object, **kwargs: object) -> None:
            raise MemoryError

        monkeypatch.setattr(av, "open", out_of_memory)
        with pytest.raises(MemoryError):
            validate_video(_make_video_bytes(num_frames=4), 10**9)


class TestFrameSubsampling:
    """The sub-sampling options each model selects, pinned in one place."""

    def test_fixed_count_floors_positions(self) -> None:
        """``rounding="floor"`` lands on the preceding frame (Gemma4)."""
        sampled = decode_video_frames(
            _make_video_bytes(num_frames=8), num_frames=4, rounding="floor"
        )
        assert sampled.indices == [0, 2, 4, 7]
        assert len(sampled.frames) == 4

    def test_fixed_count_rounds_positions(self) -> None:
        """``rounding="round"`` lands on the nearest frame (Kimi)."""
        sampled = decode_video_frames(
            _make_video_bytes(num_frames=8), num_frames=4, rounding="round"
        )
        assert sampled.indices == [0, 2, 5, 7]

    def test_requesting_more_frames_than_exist_keeps_all(self) -> None:
        sampled = decode_video_frames(
            _make_video_bytes(num_frames=3), num_frames=8
        )
        assert sampled.indices == [0, 1, 2]

    def test_target_fps_keeps_proportional_share(self) -> None:
        """2 fps out of a 24 fps clip keeps a twelfth of the frames."""
        sampled = decode_video_frames(
            _make_video_bytes(num_frames=48, rate=24), target_fps=2.0
        )
        assert len(sampled.indices) == 4
        assert sampled.indices[0] == 0
        assert sampled.indices[-1] == 47

    def test_target_fps_never_upsamples(self) -> None:
        sampled = decode_video_frames(
            _make_video_bytes(num_frames=8, rate=24), target_fps=60.0
        )
        assert len(sampled.indices) == 8

    def test_no_policy_keeps_every_frame(self) -> None:
        sampled = decode_video_frames(_make_video_bytes(num_frames=6))
        assert sampled.indices == list(range(6))

    def test_min_frames_raises_the_floor(self) -> None:
        sampled = decode_video_frames(
            _make_video_bytes(num_frames=48, rate=24),
            target_fps=0.1,
            min_frames=3,
        )
        assert len(sampled.indices) == 3

    def test_frame_range_restricts_sampling(self) -> None:
        """``start_frame``/``end_frame`` window the clip (Qwen time ranges)."""
        sampled = decode_video_frames(
            _make_video_bytes(num_frames=24),
            num_frames=3,
            start_frame=8,
            end_frame=16,
        )
        assert sampled.indices == [8, 12, 16]

    def test_num_frames_and_target_fps_are_exclusive(self) -> None:
        with pytest.raises(ValueError, match="at most one"):
            decode_video_frames(
                _make_video_bytes(num_frames=4), num_frames=2, target_fps=1.0
            )


class TestWalkSampling:
    """The walk guarantees spacing and the final frame (MiniMax-M3)."""

    def test_keeps_first_and_final_frame(self) -> None:
        """The last frame must survive: benchmarks put the question there."""
        sampled = decode_video_frames(
            _make_video_bytes(num_frames=24, rate=24),
            target_fps=1.0,
            sampling="walk",
        )
        assert sampled.indices[0] == 0
        assert sampled.indices[-1] == 23

    def test_spacing_is_at_least_one_interval(self) -> None:
        sampled = decode_video_frames(
            _make_video_bytes(num_frames=48, rate=24),
            target_fps=2.0,
            sampling="walk",
        )
        # 2 fps of a 24 fps clip means kept frames are >= 12 apart, except the
        # appended final frame which may land closer.
        gaps = np.diff(sampled.indices[:-1])
        assert np.all(gaps >= 12)

    def test_budget_thinning_keeps_the_final_frame(self) -> None:
        """Thinning an over-budget walk must not drop the final frame.

        A previous frame cap re-sampled long clips in a way that dropped the
        last frame, which broke a benchmark whose answer is that frame.
        """
        sampled = decode_video_frames(
            _make_video_bytes(num_frames=48, rate=24),
            target_fps=12.0,
            sampling="walk",
            max_decoded_bytes=FRAME_BYTES * 4,
        )
        assert len(sampled.frames) == 4
        assert sampled.indices[0] == 0
        assert sampled.indices[-1] == 47


class TestWalkIndexPlanning:
    """Walk index math, checked without decoding.

    Ported from the MiniMax-M3 sampler these replace: every case is a
    regression someone paid for once already.
    """

    def test_one_frame_per_second_plus_the_last(self) -> None:
        idx = plan_frame_indices(288, 24.0, target_fps=1.0, sampling="walk")
        assert idx[:3] == [0, 24, 48]
        assert idx[-1] == 287
        assert len(idx) == 13
        assert idx == sorted(set(idx))

    def test_last_frame_not_duplicated_when_on_grid(self) -> None:
        idx = plan_frame_indices(97, 24.0, target_fps=1.0, sampling="walk")
        assert idx == [0, 24, 48, 72, 96]

    def test_target_at_native_rate_keeps_every_frame(self) -> None:
        idx = plan_frame_indices(10, 5.0, target_fps=5.0, sampling="walk")
        assert idx == list(range(10))

    def test_sparse_walk_is_not_padded(self) -> None:
        idx = plan_frame_indices(24, 24.0, target_fps=0.2, sampling="walk")
        assert idx == [0, 23]

    def test_long_clip_is_not_capped_and_keeps_the_last_frame(self) -> None:
        idx = plan_frame_indices(100_000, 24.0, target_fps=5.0, sampling="walk")
        assert idx[:3] == [0, 5, 10]
        assert idx[-1] == 99_999
        assert idx == sorted(set(idx))
        assert len(idx) > 768

    def test_video_mmmu_long_clip_keeps_question_frame(self) -> None:
        """600 s at 30 fps sampled at 1 fps must keep the final frame.

        The question image of a Video-MMMU item is the last frame; a cap that
        dropped it collapsed the benchmark.
        """
        idx = plan_frame_indices(18_000, 30.0, target_fps=1.0, sampling="walk")
        assert idx[-1] == 17_999

    def test_single_frame_video(self) -> None:
        assert plan_frame_indices(1, 24.0, target_fps=1.0, sampling="walk") == [
            0
        ]

    def test_deterministic(self) -> None:
        first = plan_frame_indices(288, 23.976, target_fps=0.7, sampling="walk")
        second = plan_frame_indices(
            288, 23.976, target_fps=0.7, sampling="walk"
        )
        assert first == second


class TestDecodedByteBudget:
    """Decoded footprint is bounded by the same knob as everything else."""

    def test_budget_reduces_the_sampled_count(self) -> None:
        sampled = decode_video_frames(
            _make_video_bytes(num_frames=24),
            num_frames=24,
            max_decoded_bytes=FRAME_BYTES * 3,
        )
        assert len(sampled.frames) == 3

    def test_budget_beats_a_looser_model_cap(self) -> None:
        sampled = decode_video_frames(
            _make_video_bytes(num_frames=24),
            num_frames=24,
            max_frames=20,
            max_decoded_bytes=FRAME_BYTES * 2,
        )
        assert len(sampled.frames) == 2

    def test_model_cap_beats_a_looser_budget(self) -> None:
        sampled = decode_video_frames(
            _make_video_bytes(num_frames=24),
            num_frames=24,
            max_frames=5,
            max_decoded_bytes=FRAME_BYTES * 100,
        )
        assert len(sampled.frames) == 5

    def test_single_frame_over_budget_is_rejected(self) -> None:
        """When not even one frame fits there is nothing to degrade to."""
        with pytest.raises(InputError, match="more than the maximum request"):
            decode_video_frames(
                _make_video_bytes(num_frames=8),
                num_frames=1,
                max_decoded_bytes=FRAME_BYTES - 1,
            )

    def test_budget_is_computed_before_decoding(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The limit comes from the header, so it costs no pixels to apply.

        A rejected video must not convert a single frame -- that allocation is
        exactly what the budget exists to prevent.
        """
        conversions = 0
        original = av.VideoFrame.to_ndarray

        def counting(self, *args, **kwargs):  # noqa: ANN001, ANN202
            nonlocal conversions
            conversions += 1
            return original(self, *args, **kwargs)

        monkeypatch.setattr(av.VideoFrame, "to_ndarray", counting)
        with pytest.raises(InputError):
            decode_video_frames(
                _make_video_bytes(num_frames=8),
                max_decoded_bytes=FRAME_BYTES - 1,
            )
        assert conversions == 0


class TestBoundedDecode:
    """Only the sampled frames are materialized, and headers are not trusted."""

    def test_decodes_only_the_sampled_frames(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Frames outside the sample are never converted to pixels.

        Counting the conversions is the direct check that peak memory tracks
        the sampled set rather than the clip: the pre-fix implementation
        converted every frame before subsampling.
        """
        conversions = 0
        original = av.VideoFrame.to_ndarray

        def counting(self, *args, **kwargs):  # noqa: ANN001, ANN202
            nonlocal conversions
            conversions += 1
            return original(self, *args, **kwargs)

        monkeypatch.setattr(av.VideoFrame, "to_ndarray", counting)
        sampled = decode_video_frames(
            _make_video_bytes(num_frames=24), num_frames=3
        )
        assert len(sampled.frames) == 3
        assert conversions == 3

    def test_frames_are_the_ones_requested(self) -> None:
        """Source frames brighten with index, identifying which were kept."""
        sampled = decode_video_frames(
            _make_video_bytes(num_frames=24), num_frames=4
        )
        means = [float(frame.mean()) for frame in sampled.frames]
        assert np.all(np.diff(means) > 0)

    def test_missing_frame_count_uses_counting_decode(self) -> None:
        video_bytes = _make_video_bytes(
            num_frames=24, container_format="mpegts"
        )
        sampled = decode_video_frames(video_bytes, num_frames=4)
        assert len(sampled.frames) == 4
        assert sampled.info.num_frames > 0

    def test_nonzero_stream_origin_does_not_duplicate_frames(self) -> None:
        """MPEG-TS timelines start well after zero; targets are origin-relative."""
        video_bytes = _make_video_bytes(
            num_frames=24, container_format="mpegts", start_offset=90000 * 10
        )
        sampled = decode_video_frames(video_bytes, num_frames=4)
        brightness = [float(f.mean()) for f in sampled.frames]
        assert brightness == sorted(set(brightness))

    def test_over_declared_frame_count_is_corrected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A header claiming more frames than exist resamples from the truth.

        Left uncorrected, the sample is spread over a timeline that does not
        exist and most of it collapses onto the final frame.
        """
        real_probe = video_module.probe_video

        def lying_probe(source):  # noqa: ANN001, ANN202
            info = real_probe(source)
            return video_module.VideoStreamInfo(
                fps=info.fps,
                num_frames=1000,
                width=info.width,
                height=info.height,
            )

        monkeypatch.setattr(video_module, "probe_video", lying_probe)
        sampled = decode_video_frames(
            _make_video_bytes(num_frames=8), num_frames=4, rounding="floor"
        )
        assert sampled.indices == [0, 2, 4, 7]
        assert sampled.info.num_frames == 8

    def test_garbage_raises_input_error(self) -> None:
        with pytest.raises(InputError, match="invalid or unreadable video"):
            decode_video_frames(b"still not a video", num_frames=4)


class TestThreadedDecode:
    """Parallel spans must not change which frames come back."""

    def test_threaded_matches_single_threaded(self) -> None:
        video_bytes = _make_video_bytes(num_frames=64, rate=24)
        serial = decode_video_frames(video_bytes, num_frames=32, threads=1)
        parallel = decode_video_frames(video_bytes, num_frames=32, threads=4)
        assert serial.indices == parallel.indices
        for lhs, rhs in zip(serial.frames, parallel.frames, strict=True):
            np.testing.assert_array_equal(lhs, rhs)


def _make_h264_video(
    total_frames: int = 90,
    rate: int = 30,
    gop: int = 10,
    bframes: int = 0,
) -> bytes:
    """Encodes a faststart H.264 mp4 with a controllable GOP / B-frame count.

    faststart puts the header before the media data, so cutting tail bytes
    yields the shape of a truncated download: an intact header promising frames
    the data no longer contains. It needs a seekable file to rewrite on close,
    hence the temp file rather than a ``BytesIO``.
    """
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        out = av.open(path, mode="w", options={"movflags": "+faststart"})
        stream = out.add_stream("h264", rate=rate)
        assert isinstance(stream, av.VideoStream)
        stream.width = 64
        stream.height = 64
        stream.pix_fmt = "yuv420p"
        stream.options = {
            "g": str(gop),
            "crf": "23",
            "preset": "ultrafast",
            "bf": str(bframes),
        }
        for i in range(total_frames):
            img = np.zeros((64, 64, 3), dtype=np.uint8)
            img[:, :, 0] = (i * 3) % 256
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pts = i
            frame.time_base = Fraction(1, rate)
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
        out.close()
        return Path(path).read_bytes()
    finally:
        os.unlink(path)


def _make_vfr_video() -> bytes:
    """90-frame variable-rate H.264 mp4: 30 fps, then 15 fps, then 30 fps.

    The average rate (22.5 fps) matches none of the segments, so
    ``index * average_rate`` arithmetic diverges from real timestamps -- which
    is what makes this a test of timestamp-based selection.
    """
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        out = av.open(path, mode="w", options={"movflags": "+faststart"})
        stream = out.add_stream("h264", rate=30)
        assert isinstance(stream, av.VideoStream)
        stream.width = 64
        stream.height = 64
        stream.pix_fmt = "yuv420p"
        stream.options = {
            "g": "10",
            "crf": "23",
            "preset": "ultrafast",
            "bf": "0",
        }
        pts = 0
        for i in range(90):
            img = np.zeros((64, 64, 3), dtype=np.uint8)
            img[:, :, 0] = (i * 3) % 256
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pts = pts
            frame.time_base = Fraction(1, 30)
            pts += 2 if 30 <= i < 60 else 1
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
        out.close()
        return Path(path).read_bytes()
    finally:
        os.unlink(path)


def _decode_all(video_bytes: bytes) -> list[npt.NDArray[np.uint8]]:
    """Full decode, as the reference the sampled decode must agree with."""
    with open_video_container(io.BytesIO(video_bytes)) as container:
        return [
            frame.to_ndarray(format="rgb24").astype(np.uint8, copy=False)
            for frame in container.decode(video=0)
        ]


class TestDamagedAndIrregularStreams:
    """Selecting by timestamp is what makes these cases come out right.

    Every one of these was a real failure of the index-counting decoder these
    tests' subject replaced.
    """

    def test_vfr_selects_by_timestamp(self) -> None:
        """On variable-rate video the sampled frames must be the ones whose
        real timestamps match the target times, not the ones an index-count
        walk lands on."""
        video_bytes = _make_vfr_video()
        ref_frames: list[npt.NDArray[np.uint8]] = []
        ref_times: list[float] = []
        with open_video_container(io.BytesIO(video_bytes)) as container:
            stream = container.streams.video[0]
            assert stream.average_rate is not None
            assert stream.time_base is not None
            native_fps = float(stream.average_rate)
            time_base = stream.time_base
            for frame in container.decode(stream):
                assert frame.pts is not None
                ref_frames.append(
                    frame.to_ndarray(format="rgb24").astype(
                        np.uint8, copy=False
                    )
                )
                ref_times.append(float(frame.pts * time_base))
        assert native_fps != 30.0  # fixture sanity: genuinely variable-rate

        sampled = decode_video_frames(
            video_bytes, target_fps=1.0, sampling="walk", threads=2
        )
        half = 0.5 / native_fps
        for position, index in enumerate(sampled.indices):
            target = index / native_fps
            expected = next(
                j for j, t in enumerate(ref_times) if t >= target - half
            )
            np.testing.assert_array_equal(
                sampled.frames[position],
                ref_frames[expected],
                err_msg=(
                    f"sample {position} (target {target:.3f}s) is not the"
                    " frame whose timestamp matches"
                ),
            )

    def test_bframe_clip_matches_full_decode(self) -> None:
        """B-frames reorder decoder output and the gap accelerator skips
        non-reference frames; the sample must still match a full decode."""
        video_bytes = _make_h264_video(total_frames=90, gop=30, bframes=2)
        reference = _decode_all(video_bytes)
        sampled = decode_video_frames(
            video_bytes, target_fps=1.0, sampling="walk", threads=2
        )
        for position, index in enumerate(sampled.indices):
            np.testing.assert_array_equal(
                sampled.frames[position], reference[index]
            )

    def test_truncated_download_substitutes_rather_than_crashing(self) -> None:
        """A truncated download keeps a header promising frames the data no
        longer contains. The decode must serve the nearest decodable frame
        instead of failing the request."""
        full = _make_h264_video(total_frames=300, rate=30, gop=10)
        truncated = full[: int(len(full) * 0.9)]
        # Fixture sanity: the header still claims the full frame count.
        with open_video_container(io.BytesIO(truncated)) as container:
            assert container.streams.video[0].frames == 300

        sampled = decode_video_frames(truncated, num_frames=8)
        assert len(sampled.frames) == 8

    def test_indices_past_the_stream_end_are_substituted(self) -> None:
        """Sampled positions beyond the real stream keep the output shape."""
        video_bytes = _make_h264_video(total_frames=90)
        reference = _decode_all(video_bytes)
        out, missing = video_module._decode_at_indices(
            video_bytes, [0, 5, 300], 30.0, 1
        )
        assert len(out) == 3
        assert missing == 1
        np.testing.assert_array_equal(out[2], reference[len(reference) - 1])

    def test_temp_file_is_cleaned_up(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The parallel decode stages bytes in one temp file; it must be
        removed on both the success and the failure path."""
        created: list[str] = []
        real_mkstemp = tempfile.mkstemp

        def tracking_mkstemp(*args, **kwargs):  # noqa: ANN202
            fd, path = real_mkstemp(*args, **kwargs)
            created.append(path)
            return fd, path

        monkeypatch.setattr(video_module.tempfile, "mkstemp", tracking_mkstemp)

        video_bytes = _make_h264_video(total_frames=90)
        decode_video_frames(video_bytes, num_frames=32, threads=4)
        assert created and not any(os.path.exists(p) for p in created)

        created.clear()
        with pytest.raises(InputError):
            decode_video_frames(b"not a video", num_frames=32, threads=4)
        assert not any(os.path.exists(p) for p in created)


class TestFrameConversion:
    """The identity that lets one internal representation serve every caller."""

    def test_ndarray_is_pixel_identical_to_pil(self) -> None:
        """``to_ndarray("rgb24")`` must equal ``to_image().convert("RGB")``.

        Callers that want images get a cheap wrapper around the decoded array
        rather than a second conversion; that is only sound while these agree.
        """
        video_bytes = _make_video_bytes(num_frames=8)
        with open_video_container(io.BytesIO(video_bytes)) as container:
            for frame in container.decode(video=0):
                np.testing.assert_array_equal(
                    np.asarray(frame.to_image().convert("RGB")),
                    frame.to_ndarray(format="rgb24"),
                )

    def test_images_and_array_views_agree(self) -> None:
        sampled = decode_video_frames(
            _make_video_bytes(num_frames=8), num_frames=4
        )
        images = sampled.images
        assert len(images) == 4
        assert sampled.array.shape == (4, HEIGHT, WIDTH, 3)
        for image, frame in zip(images, sampled.frames, strict=True):
            np.testing.assert_array_equal(np.asarray(image), frame)
