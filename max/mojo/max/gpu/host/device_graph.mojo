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
"""This module provides functionality for building and replaying device graphs.
A device graph captures a sequence of GPU operations (such as kernel launches,
memory copies, and memsets) as a reusable graph that can be replayed at a lower
overhead than re-enqueueing each operation individually. The main entry point
is [`DeviceGraph.create()`](/api/mojo/max/gpu/host/device_graph/DeviceGraph/#create),
which hands a [`DeviceGraphBuilder`](/api/mojo/max/gpu/host/device_graph/DeviceGraphBuilder/)
to a scoped callback.

Graph capture is currently implemented for CUDA and HIP devices only.
Creating a graph on any other device, such as an Apple GPU or a CPU, raises."""


@__doc_inline
from ._overlay import (
    DeviceGraph,
    DeviceGraphBuilder,
    DeviceGraphCache,
    DeviceGraphInput,
    DeviceGraphInputKind,
    DeviceGraphNode,
    StableAddr,
)
