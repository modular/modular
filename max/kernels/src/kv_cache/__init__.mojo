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
"""
Contains implementations for several types of key-value caches.

[KV caches](/glossary/ai/kv-cache) are used in transformer models to store
key-value tensors output from self-attention layers.

These APIs are used in the higher-level functions in the
[`nn`](/mojo/kernels/nn) package.
"""
