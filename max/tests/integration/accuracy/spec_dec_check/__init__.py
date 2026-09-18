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
"""SpecDecCheck — statistical losslessness check for speculative decoding.

Speculative decoding is meant to be distribution-preserving: the tokens a
server emits with drafting enabled should come from the same distribution as
the tokens the target model emits on its own. SpecDecCheck tests that claim
empirically, by drawing many seeded completions from a server with drafting
enabled and a server without it and comparing the per-position token
distributions, so a drafting bug that shifts the distribution shows up as a
significant difference between the servers rather than as a single mismatched
completion.
"""
