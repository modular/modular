# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

"""Pixel generation scheduler utilities for image and video generation metrics."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from enum import Enum

from max.serve.telemetry.metrics import METRICS
from max.support.human_readable_formatter import to_human_readable_latency

logger = logging.getLogger("max.serve")


class PixelBatchType(str, Enum):
    """Type of pixel generation batch."""

    IMAGE = "IMG"
    """Image generation batch."""

    VIDEO = "VID"
    """Video generation batch."""


def _to_human_readable_pixel_throughput(value: float, unit: str) -> str:
    """Format throughput values for pixel generation.

    Args:
        value: The throughput value.
        unit: The unit string (e.g., "step/s", "img/s", "frame/s").

    Returns:
        Formatted string with appropriate scaling.
    """
    if value >= 1_000:
        return f"{value / 1e3:.2f}K {unit}"
    return f"{value:.2f} {unit}"


@dataclass
class PixelBatchMetrics:
    """Metrics for a pixel generation batch (image or video).

    This class captures performance metrics for diffusion model inference,
    including resolution, inference steps, and throughput measurements.
    """

    batch_type: PixelBatchType
    """Whether this is an image or video generation batch."""

    batch_size: int
    """Number of requests in this batch."""

    num_images_per_prompt: int
    """Number of images/videos generated per request."""

    height: int
    """Output height in pixels."""

    width: int
    """Output width in pixels."""

    num_inference_steps: int
    """Number of diffusion denoising steps."""

    num_frames: int | None
    """Number of frames for video generation, None for images."""

    terminated_reqs: int
    """Number of requests that completed in this batch."""

    num_pending_reqs: int
    """Number of requests waiting in queue."""

    batch_creation_time_s: float
    """Time to construct the batch in seconds."""

    batch_execution_time_s: float
    """Time to execute the full diffusion inference in seconds."""

    @property
    def total_outputs(self) -> int:
        """Total number of images/videos generated."""
        return self.batch_size * self.num_images_per_prompt

    @property
    def steps_per_second(self) -> float:
        """Diffusion steps executed per second."""
        if self.batch_execution_time_s == 0:
            return 0.0
        total_steps = self.num_inference_steps * self.total_outputs
        return total_steps / self.batch_execution_time_s

    @property
    def outputs_per_second(self) -> float:
        """Images or videos generated per second."""
        if self.batch_execution_time_s == 0:
            return 0.0
        return self.total_outputs / self.batch_execution_time_s

    @property
    def latency_per_output_s(self) -> float:
        """Average latency per output in seconds."""
        if self.total_outputs == 0:
            return 0.0
        return self.batch_execution_time_s / self.total_outputs

    @property
    def megapixels(self) -> float:
        """Total megapixels generated."""
        pixels = self.height * self.width * self.total_outputs
        if self.num_frames:
            pixels *= self.num_frames
        return pixels / 1_000_000

    def pretty_format(self) -> str:
        """Format metrics as a human-readable log string.

        Returns:
            Formatted string matching Modular's logging style.
        """
        # Resolution string
        if self.num_frames:
            resolution_str = f"{self.height}x{self.width}x{self.num_frames}f"
            output_unit = "vid/s"
        else:
            resolution_str = f"{self.height}x{self.width}"
            output_unit = "img/s"

        # Throughput strings
        steps_tput_str = _to_human_readable_pixel_throughput(
            self.steps_per_second, "step/s"
        )
        output_tput_str = _to_human_readable_pixel_throughput(
            self.outputs_per_second, output_unit
        )

        return (
            f"Executed {self.batch_type.value} batch with {self.batch_size} reqs | "
            f"Terminated: {self.terminated_reqs} reqs, "
            f"Pending: {self.num_pending_reqs} reqs | "
            f"Size: {resolution_str}, Steps: {self.num_inference_steps} | "
            f"Step Tput: {steps_tput_str}, "
            f"Output Tput: {output_tput_str} | "
            f"Batch creation: {to_human_readable_latency(self.batch_creation_time_s)}, "
            f"Execution: {to_human_readable_latency(self.batch_execution_time_s)}"
        )

    def publish_metrics(self) -> None:
        """Publish metrics to the telemetry system."""
        METRICS.batch_size(self.batch_size)
        METRICS.batch_execution_time(
            self.batch_execution_time_s * 1000,  # Convert to ms
            batch_type=self.batch_type.value,
        )
        # TODO: Add pixel-specific metrics when METRICS is extended
        # METRICS.pixel_outputs_per_second(self.outputs_per_second)
        # METRICS.pixel_steps_per_second(self.steps_per_second)


class PixelSchedulerLogger:
    """Class to periodically log pixel generation batch metrics to console."""

    def __init__(self, log_interval_s: float | None = None) -> None:
        """Initializes the PixelSchedulerLogger.

        Args:
            log_interval_s: How frequently to log batches, in seconds.
        """
        if log_interval_s is None:
            log_interval_s = float(
                os.getenv("MAX_SERVE_PIXEL_SCHEDULER_STATS_LOG_INTERVAL_S", "3")
            )
        logger.debug(
            f"Enabled pixel scheduler batch statistic logging at interval of {log_interval_s:.2f}s"
        )

        self.log_interval_s = log_interval_s
        self.time_of_last_log = 0.0

    def log_metrics(
        self,
        batch_type: PixelBatchType,
        batch_size: int,
        num_images_per_prompt: int,
        height: int,
        width: int,
        num_inference_steps: int,
        num_frames: int | None,
        batch_creation_time_s: float,
        batch_execution_time_s: float,
        num_pending_reqs: int,
        num_terminated_reqs: int,
    ) -> None:
        """Log pixel generation batch metrics.

        Args:
            batch_type: Whether this is image or video generation.
            batch_size: Number of requests in the batch.
            num_images_per_prompt: Number of outputs per request.
            height: Output height in pixels.
            width: Output width in pixels.
            num_inference_steps: Number of diffusion steps.
            num_frames: Number of frames (video only).
            batch_creation_time_s: Time to create the batch.
            batch_execution_time_s: Time to execute the batch.
            num_pending_reqs: Number of pending requests.
            num_terminated_reqs: Number of terminated requests.
        """
        metrics = PixelBatchMetrics(
            batch_type=batch_type,
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            terminated_reqs=num_terminated_reqs,
            num_pending_reqs=num_pending_reqs,
            batch_creation_time_s=batch_creation_time_s,
            batch_execution_time_s=batch_execution_time_s,
        )

        # Always publish metrics
        metrics.publish_metrics()

        # Only periodically log to console to avoid spam
        now = time.monotonic()
        time_since_last_log = now - self.time_of_last_log
        if self.log_interval_s < time_since_last_log:
            self.time_of_last_log = now
            logger.info(metrics.pretty_format())
