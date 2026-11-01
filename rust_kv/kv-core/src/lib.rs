//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//! Host-side KV cache bookkeeping shared by MAX Serve and Mach.
//!
//! `kv-core` is meant to become the one KV cache manager behind both engines,
//! so a prefix-caching or hybrid-model rule is written and fixed once. It is
//! sans-IO: no GPU, async runtime, or Python dependency. The engines own
//! buffers and device copies; this crate decides what can be reused.
//!
//! It starts with the rules that were already engine-agnostic, in
//! [`prefix_hit`], the Jenga block pool that every flat KV cache draws its
//! blocks from, in [`pool`], and the geometry math that sizes and tiles the
//! Jenga slab, in [`geometry`]. The chained block hashes that key the prefix
//! cache are in [`hash`].

pub mod geometry;
pub mod hash;
pub mod pool;
pub mod prefix_hit;
