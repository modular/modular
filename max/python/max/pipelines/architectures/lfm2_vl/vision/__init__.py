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
"""The LFM2-VL vision side: tower plus fold projector, one graph.

Import the submodule you need: `vision.tower` builds the graph,
`vision.attention` is the per-tile bidirectional MHSA, and
`vision.projector` folds 2x2 patch blocks and projects them.
"""
