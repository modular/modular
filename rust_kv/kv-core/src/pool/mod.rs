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
//! The fungible huge-block pool every flat KV cache draws from.
//!
//! MAX's device pool runs on this through `max._kv_core.JengaBlockPool`
//! (`rust_kv/kv-core-pyo3/src/pool.rs`); the Python pool it was ported from,
//! `jenga_block_pool.py` and its huge/little block classes in
//! `block_utils.py`, is gone. The kv tier connector's host pool runs on it
//! too. Unlike one
//! flat run of equally sized blocks (the single-cache case,
//! `cache_ratios.len() == 1`), several caches with different page sizes tile
//! the same bytes here, and the split between them follows live demand rather
//! than a knob.
//!
//! Three things differ from the deleted Python pool:
//!
//! - The null block is optional. MAX's device pool always reserves huge block
//!   0, and block 0 of every cache, for the block that dummy and padding
//!   requests share. [`block_pool::PoolOptions::null_block`] does the same; a
//!   host tier has no dummy requests, so it leaves the option off, every huge
//!   block is allocatable, ids start at 0, and one huge block is a pool.
//! - A block is addressed by the `(cache, bid)` pair that the Python block
//!   object carries in its fields, and a `bid` is an index into that cache's
//!   `little_blocks`, so the block itself needs no back-pointers. Huge block 0
//!   stays in those arrays when it holds the null block, so a `bid` is its page
//!   index into the cache's view of the buffer either way.
//! - A huge block tracks how many of its little blocks are referenced, kept
//!   current on the 0-to-1 transitions. Python recomputes the sum of their
//!   reference counts on every read, which is O(ratio) per free -- and a ratio
//!   reaches the hundreds or thousands for a cache of small pages. Only the
//!   count being zero is ever asked, and the two agree on that.
//!
//! Single-threaded: its owner serializes every mutation (the tiered connector
//! does all of them on its main thread, and MAX's binding holds the pool
//! behind a mutex), so nothing in here locks.

pub mod block;
pub mod block_pool;
pub mod free_list;
