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
"""Tests for the roi_align graph op."""

import pytest
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops


class TestRoiAlignGraphOp:
    """Tests that ops.roi_align builds correct graph types and shapes."""

    def test_basic_shape(self) -> None:
        """Output shape is [num_rois, out_h, out_w, channels]."""
        input_type = TensorType(DType.float32, [1, 10, 10, 3], DeviceRef.CPU())
        rois_type = TensorType(DType.float32, [4, 5], DeviceRef.CPU())

        with Graph(
            "roi_align_basic", input_types=[input_type, rois_type]
        ) as graph:
            x, rois = graph.inputs[0].tensor, graph.inputs[1].tensor
            out = ops.roi_align(x, rois, output_height=3, output_width=3)
            graph.output(out)

            assert out.type.dtype == DType.float32
            assert list(out.type.shape) == [4, 3, 3, 3]

    def test_preserves_dtype(self) -> None:
        """Output dtype matches input dtype."""
        input_type = TensorType(DType.float64, [2, 8, 8, 1], DeviceRef.CPU())
        rois_type = TensorType(DType.float64, [1, 5], DeviceRef.CPU())

        with Graph(
            "roi_align_dtype", input_types=[input_type, rois_type]
        ) as graph:
            x, rois = graph.inputs[0].tensor, graph.inputs[1].tensor
            out = ops.roi_align(x, rois, output_height=2, output_width=2)
            graph.output(out)

            assert out.type.dtype == DType.float64

    def test_max_mode(self) -> None:
        """Graph builds correctly with mode='MAX'."""
        input_type = TensorType(DType.float32, [1, 10, 10, 1], DeviceRef.CPU())
        rois_type = TensorType(DType.float32, [2, 5], DeviceRef.CPU())

        with Graph(
            "roi_align_max", input_types=[input_type, rois_type]
        ) as graph:
            x, rois = graph.inputs[0].tensor, graph.inputs[1].tensor
            out = ops.roi_align(
                x, rois, output_height=4, output_width=4, mode="MAX"
            )
            graph.output(out)

            assert list(out.type.shape) == [2, 4, 4, 1]

    def test_aligned_flag(self) -> None:
        """Graph builds correctly with aligned=True."""
        input_type = TensorType(DType.float32, [1, 5, 5, 2], DeviceRef.CPU())
        rois_type = TensorType(DType.float32, [3, 5], DeviceRef.CPU())

        with Graph(
            "roi_align_aligned", input_types=[input_type, rois_type]
        ) as graph:
            x, rois = graph.inputs[0].tensor, graph.inputs[1].tensor
            out = ops.roi_align(
                x, rois, output_height=2, output_width=2, aligned=True
            )
            graph.output(out)

            assert list(out.type.shape) == [3, 2, 2, 2]

    def test_invalid_input_rank(self) -> None:
        """Raises ValueError for non-rank-4 input."""
        input_type = TensorType(DType.float32, [10, 10], DeviceRef.CPU())
        rois_type = TensorType(DType.float32, [1, 5], DeviceRef.CPU())

        with Graph(
            "roi_align_bad_rank", input_types=[input_type, rois_type]
        ) as graph:
            x, rois = graph.inputs[0].tensor, graph.inputs[1].tensor
            with pytest.raises(ValueError, match="rank-4"):
                ops.roi_align(x, rois, output_height=3, output_width=3)

    def test_invalid_rois_rank(self) -> None:
        """Raises ValueError for non-rank-2 rois."""
        input_type = TensorType(DType.float32, [1, 10, 10, 1], DeviceRef.CPU())
        rois_type = TensorType(DType.float32, [1, 5, 1], DeviceRef.CPU())

        with Graph(
            "roi_align_bad_rois", input_types=[input_type, rois_type]
        ) as graph:
            x, rois = graph.inputs[0].tensor, graph.inputs[1].tensor
            with pytest.raises(ValueError, match="rank-2"):
                ops.roi_align(x, rois, output_height=3, output_width=3)

    def test_invalid_mode(self) -> None:
        """Raises ValueError for invalid mode."""
        input_type = TensorType(DType.float32, [1, 10, 10, 1], DeviceRef.CPU())
        rois_type = TensorType(DType.float32, [1, 5], DeviceRef.CPU())

        with Graph(
            "roi_align_bad_mode", input_types=[input_type, rois_type]
        ) as graph:
            x, rois = graph.inputs[0].tensor, graph.inputs[1].tensor
            with pytest.raises(ValueError, match="mode"):
                ops.roi_align(
                    x, rois, output_height=3, output_width=3, mode="SUM"
                )
