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
//! `#[pyclass]` wrappers over `kv_tier_connector::connector::TieredConnector`.
//!
//! Device buffers and the pinned host buffer cross the boundary as raw
//! addresses (ints), exactly as the dKV connector does. Every blocking call
//! releases the GIL with `Python::detach` so the connector's lanes (Rust OS
//! threads) run without contending for it.

use std::{collections::HashMap, sync::Arc, time::Duration};

use kv_tier_connector::{
    connector::{LeafRecency, TieredConnector, TransferResult},
    copy_engine::{CopyEngine, LeafMem, Unit},
    disk_tier::DiskTier,
    host_region::HostRegion,
    transfer::Counter,
    utils::BlockHash,
};
use pyo3::{
    PyErr,
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};

/// Default number of disk-tier worker threads (reads/deletes/writes share this
/// pool). Overridable per-run via the `num_disk_workers` connector config
/// field.
const DISK_WORKERS: usize = 32;

/// A `(device_id, data_ptr, bytes_per_page)` unit as passed from Python --
/// already the per-block stride, not the shard's full buffer size.
type PyUnit = (u64, usize, usize);
/// A leaf's device endpoints: `(units, replicated)`.
type PyLeaf = (Vec<PyUnit>, bool);

/// Handle for one in-flight (or completed) transfer; the scheduler polls
/// `is_complete`. `g0_blocks_per_leaf` is what an offload chose to take, which
/// the caller pins until the copy lands; a load is handed its blocks and so
/// only restates them.
#[pyclass(module = "kv_tier_connector")]
pub(crate) struct TierTransfer {
    counter: Counter,
    #[pyo3(get)]
    g0_blocks_per_leaf: Vec<Vec<usize>>,
    /// Blocks an offload skipped because the host tier already held them.
    #[pyo3(get)]
    already_held: usize,
}

#[pymethods]
impl TierTransfer {
    fn is_complete(&self) -> bool {
        self.counter.is_complete()
    }

    /// Blocks until complete (drain/shutdown only).
    fn synchronize(&self, py: Python<'_>) {
        py.detach(|| {
            while !self.counter.is_complete() {
                std::thread::sleep(Duration::from_micros(200));
            }
        });
    }
}

/// Converts a `[(units, replicated)]` per replica, per leaf, into the
/// engine's `LeafMem` tree.
fn replicas_from(py_replicas: Vec<Vec<PyLeaf>>) -> Vec<Vec<LeafMem>> {
    py_replicas
        .into_iter()
        .map(|leaves| {
            leaves
                .into_iter()
                .map(|(units, replicated)| LeafMem {
                    units: units
                        .into_iter()
                        .map(|(device_id, ptr, bytes)| Unit {
                            device_id,
                            ptr,
                            bytes,
                        })
                        .collect(),
                    replicated,
                })
                .collect()
        })
        .collect()
}

/// Maps a connector error onto the Python exception its cause deserves.
///
/// A `Config` error means the caller handed us mismatched shapes, which is a
/// bug in the cache manager; a `TransferFailed` means a hash we were asked for
/// is no longer resident, which is a race the caller is expected to absorb as
/// a cache miss. Keeping them apart is what lets the shim turn one into a miss
/// without swallowing the other.
fn connector_error(error: kv_tier_connector::error::ConnectorError) -> PyErr {
    match error {
        kv_tier_connector::error::ConnectorError::Config(message) => PyValueError::new_err(message),
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

/// Converts raw `kv_hash_algo` bytes (8 or 32 bytes) into the connector's
/// canonical hash, erroring out on a malformed length instead of panicking.
fn hashes_from(block_hashes: &[Vec<u8>]) -> PyResult<Vec<BlockHash>> {
    block_hashes
        .iter()
        .map(|bytes| BlockHash::try_from(bytes.as_slice()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// The Python-facing tiered (or local) KV connector.
#[pyclass(module = "kv_tier_connector", unsendable)]
pub(crate) struct TierConnector {
    inner: TieredConnector,
}

#[allow(
    clippy::needless_pass_by_value,
    reason = "pyo3 extracts Python objects into owned Rust values at the method boundary"
)]
#[pymethods]
impl TierConnector {
    /// Builds the tiered connector with `disk_dir` backing the disk last
    /// level; `disk_dir=None` builds a host-only connector with no disk tier.
    ///
    /// The pinned host region arrives as `host_chunk_bases`, each chunk
    /// holding `host_rows_per_chunk` huge blocks (the last may hold fewer). A
    /// contiguous region is one base with `host_offload_num_huge_blocks` rows.
    ///
    /// `use_disk_direct_io` opts the disk tier into O_DIRECT with no buffered
    /// fallback; when true, construction first proves it end to end -- one
    /// direct write and read through the *staging region itself* into the
    /// cache dir -- and raises `ValueError` naming the failing piece
    /// (filesystem without O_DIRECT support, unaligned block sizes, or a
    /// staging region that rejects it, like driver-allocated GTT memory).
    #[new]
    #[allow(
        clippy::too_many_arguments,
        reason = "connector construction threads through all tier/replica config"
    )]
    #[pyo3(signature = (
        bytes_per_leaf,
        host_offload_num_huge_blocks,
        host_offload_cache_ratios,
        host_chunk_bases,
        host_rows_per_chunk,
        replica_kv_memory,
        device_to_stream,
        only_last_level,
        disk_dir,
        disk_offload_max_bytes,
        num_disk_workers=DISK_WORKERS,
        num_h2d_onload_lanes=1,
        use_disk_direct_io=false,
        *,
        leaf_recency,
    ))]
    fn new(
        py: Python<'_>,
        bytes_per_leaf: Vec<usize>,
        host_offload_num_huge_blocks: usize,
        host_offload_cache_ratios: Vec<usize>,
        host_chunk_bases: Vec<usize>,
        host_rows_per_chunk: usize,
        replica_kv_memory: Vec<Vec<PyLeaf>>,
        device_to_stream: HashMap<u64, u64>,
        only_last_level: bool,
        disk_dir: Option<String>,
        disk_offload_max_bytes: usize,
        num_disk_workers: usize,
        num_h2d_onload_lanes: usize,
        use_disk_direct_io: bool,
        leaf_recency: Vec<(String, usize)>,
    ) -> PyResult<Self> {
        let recency = leaf_recency
            .iter()
            .map(|(kind, blocks)| match kind.as_str() {
                "full_sequence" => Ok(LeafRecency::FullSequence),
                "sliding_window" => Ok(LeafRecency::SlidingWindow { blocks: *blocks }),
                "recurrent" => Ok(LeafRecency::Recurrent),
                other => Err(PyValueError::new_err(format!(
                    "unknown leaf recency {other:?}; expected full_sequence, \
                     sliding_window or recurrent"
                ))),
            })
            .collect::<PyResult<Vec<_>>>()?;
        if replica_kv_memory.is_empty() || replica_kv_memory[0].is_empty() {
            return Err(PyValueError::new_err(
                "replica_kv_memory must have at least one replica, each with at least one leaf",
            ));
        }
        for mem in replica_kv_memory.iter() {
            if mem.len() != replica_kv_memory[0].len() {
                return Err(PyValueError::new_err(
                    "all replicas must have the same number of leaves",
                ));
            }
        }
        let host = HostRegion::new(
            host_chunk_bases,
            host_rows_per_chunk,
            host_offload_num_huge_blocks,
            &bytes_per_leaf,
            &host_offload_cache_ratios,
        )
        .map_err(connector_error)?;

        let replicas = replicas_from(replica_kv_memory);
        // Every leaf shares one replica's device ordering (TP doesn't reshard
        // across leaves), so leaf 0 gives the per-replica device list
        // `CopyEngine` groups its aux streams by.
        let device_ids: Vec<u64> = replicas
            .iter()
            .flat_map(|leaves| leaves[0].units.iter().map(|u| u.device_id))
            .collect();

        let disk = Arc::new(match disk_dir {
            Some(dir) => {
                if use_disk_direct_io {
                    // Prove direct I/O through the real staging memory before
                    // building anything: one DIRECT_IO_ALIGN write + read at
                    // the last chunk's base (host blocks hold no valid data
                    // until an offload fills them, and construction runs
                    // before any offload) into the cache dir. Driver-allocated
                    // GTT staging (AMD) rejects O_DIRECT with EFAULT -- the
                    // exact conflict this catches; a heap buffer would miss it.
                    #[allow(
                        unsafe_code,
                        reason = "the probe takes a raw staging address; see the SAFETY comment below"
                    )]
                    py.detach(|| {
                        // Reject geometry the probe could run off the end of
                        // before touching staging memory: every leaf size and
                        // every chunk base must be direct-I/O aligned.
                        kv_tier_connector::disk_tier::check_direct_io_leaf_sizes(&bytes_per_leaf)?;
                        host.check_direct_io()?;
                        let last = *host.chunk_bases().last().ok_or_else(|| {
                            kv_tier_connector::error::ConnectorError::Config(
                                "the host region has no chunks".to_owned(),
                            )
                        })?;
                        // SAFETY: after the checks above, `last` is the aligned
                        // base of a chunk at least DIRECT_IO_ALIGN bytes long, in
                        // `host`, which outlives the call. No offload has started,
                        // so nothing else touches the span.
                        unsafe {
                            kv_tier_connector::disk_tier::probe_direct_io_at(
                                last,
                                std::path::Path::new(&dir),
                            )
                        }
                    })
                    .map_err(connector_error)?;
                }
                py.detach(|| {
                    DiskTier::new(
                        &dir,
                        &bytes_per_leaf,
                        disk_offload_max_bytes,
                        num_disk_workers,
                        use_disk_direct_io,
                    )
                })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            }
            None => DiskTier::disabled(),
        });

        let engine = Arc::new(
            py.detach(|| {
                CopyEngine::new(
                    host,
                    &bytes_per_leaf,
                    replicas,
                    &device_ids,
                    &device_to_stream,
                    num_h2d_onload_lanes,
                )
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        );

        let inner = py
            .detach(|| {
                TieredConnector::new(
                    host_offload_num_huge_blocks,
                    host_offload_cache_ratios,
                    disk,
                    engine,
                    bytes_per_leaf,
                    only_last_level,
                    num_h2d_onload_lanes,
                    recency,
                )
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(TierConnector { inner })
    }

    /// Which of `block_hashes` each of `leaf_idxs` holds, one mask per leaf.
    ///
    /// Presence only -- one call covers the whole cache tree, and turning the
    /// answers into a usable prefix is the cache manager's job. A `true` says
    /// the block is readable for *that leaf*; it makes no claim about the
    /// leaf's siblings, and it does not pin, so `load` may still find it gone.
    ///
    /// Group-major so each hash crosses the boundary once rather than once per
    /// leaf, matching the dKV client's `lookup`.
    fn lookup(
        &self,
        py: Python<'_>,
        leaf_idxs: Vec<usize>,
        block_hashes: Vec<Vec<u8>>,
    ) -> PyResult<Vec<Vec<bool>>> {
        let block_hashes = hashes_from(&block_hashes)?;
        Ok(py.detach(|| self.inner.lookup(&leaf_idxs, &block_hashes)))
    }

    /// Loads `hashes_per_leaf[leaf_idx]` into `block_ids[leaf_idx]`, in order.
    ///
    /// Raises if a requested hash is resident in neither tier.
    #[pyo3(signature = (block_ids, hashes_per_leaf, replica_idx=0))]
    fn load(
        &mut self,
        py: Python<'_>,
        block_ids: Vec<Vec<usize>>,
        hashes_per_leaf: Vec<Vec<Vec<u8>>>,
        replica_idx: usize,
    ) -> PyResult<TierTransfer> {
        let hashes_per_leaf = hashes_per_leaf
            .iter()
            .map(|hashes| hashes_from(hashes))
            .collect::<PyResult<Vec<_>>>()?;
        let TransferResult {
            g0_blocks_per_leaf,
            counter,
            already_held,
        } = py
            .detach(|| self.inner.load(&block_ids, &hashes_per_leaf, replica_idx))
            .map_err(connector_error)?;
        Ok(TierTransfer {
            counter,
            g0_blocks_per_leaf,
            already_held,
        })
    }

    /// Offloads `block_ids[leaf_idx]` under `hashes_per_leaf[leaf_idx]`, in
    /// order. A leaf's two lists are the same length; leaves need not agree.
    ///
    /// Raises `ValueError` when they do not line up, a caller bug rather than
    /// anything the tiers did.
    #[pyo3(signature = (block_ids, hashes_per_leaf, replica_idx=0))]
    fn offload(
        &mut self,
        py: Python<'_>,
        block_ids: Vec<Vec<usize>>,
        hashes_per_leaf: Vec<Vec<Vec<u8>>>,
        replica_idx: usize,
    ) -> PyResult<TierTransfer> {
        let hashes_per_leaf = hashes_per_leaf
            .iter()
            .map(|hashes| hashes_from(hashes))
            .collect::<PyResult<Vec<_>>>()?;
        let TransferResult {
            g0_blocks_per_leaf,
            counter,
            already_held,
        } = py
            .detach(|| {
                self.inner
                    .offload(&block_ids, &hashes_per_leaf, replica_idx)
            })
            .map_err(connector_error)?;
        Ok(TierTransfer {
            counter,
            g0_blocks_per_leaf,
            already_held,
        })
    }

    /// Refreshes each leaf's recency for the blocks of `block_hashes` it reads.
    fn touch(&mut self, py: Python<'_>, block_hashes: Vec<Vec<u8>>) -> PyResult<()> {
        let block_hashes = hashes_from(&block_hashes)?;
        py.detach(|| self.inner.touch(&block_hashes));
        Ok(())
    }

    fn reclaim(&mut self) {
        self.inner.reclaim();
    }

    /// Blocks until all in-flight transfers (D2H/H2D, disk read/write) drain,
    /// without clearing any tier. Releases the GIL while the lanes finish.
    fn wait_for_writes(&mut self, py: Python<'_>) {
        py.detach(|| self.inner.wait_for_writes());
    }

    fn host_bytes(&self) -> u64 {
        self.inner.host_bytes()
    }
    fn free_host_bytes(&self) -> u64 {
        self.inner.free_host_bytes()
    }
    fn disk_bytes(&self) -> u64 {
        self.inner.disk_bytes()
    }
    fn free_disk_bytes(&self) -> u64 {
        self.inner.free_disk_bytes()
    }

    /// `(h2d, d2h, disk_read, disk_write)` byte counts since the last reset.
    fn metrics(&self) -> (u64, u64, u64, u64) {
        self.inner.metrics()
    }
    fn inflight_disk_ops(&self) -> u64 {
        self.inner.inflight_disk_ops()
    }
    /// `(h2d, d2h, disk_read, disk_write)` byte counts read and cleared in one
    /// step.
    fn take_metrics(&self) -> (u64, u64, u64, u64) {
        self.inner.take_metrics()
    }
    fn reset_prefix_cache(&mut self) {
        self.inner.reset_prefix_cache();
    }
    fn shutdown(&mut self, py: Python<'_>) {
        py.detach(|| self.inner.shutdown());
    }
}
