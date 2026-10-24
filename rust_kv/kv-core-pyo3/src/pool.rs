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
//! MAX's device Jenga block pool, backed by [`kv_core::pool`].
//!
//! MAX's coordinators and block manager still walk a request's blocks one at a
//! time, so this keeps the Python pool's per-block API: blocks are objects, a
//! cache is named by its leaf id, and a hash is `bytes`. That costs one
//! crossing per block operation, which goes away once the coordinators move to
//! `kv-core` as well.
//!
//! A block object is a handle on a `(cache, bid)` pair of the pool, and the
//! pool hands out the same object for a pair every time: the coordinators
//! compare blocks with `is`. Handles reach the pool's state through a weak
//! reference, so they do not keep a pool alive and no Python reference cycle
//! forms between a pool and its blocks.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex, MutexGuard, Weak},
};

use kv_core::pool::block_pool::{JengaBlockPool, JengaError, PoolOptions};
use pyo3::{
    create_exception,
    exceptions::{PyException, PyIndexError, PyKeyError, PyRuntimeError, PyValueError},
    prelude::*,
    sync::MutexExt,
    types::{PyBytes, PyDict, PyList, PyString},
};

create_exception!(
    max._kv_core,
    InsufficientBlocksError,
    PyException,
    "Exception raised when there are insufficient free blocks to satisfy an allocation."
);

/// The longest block hash MAX produces: a SHA-256 digest. `ahash64` and
/// `sha256_64` hashes are 8 bytes.
const MAX_HASH_BYTES: usize = 32;

/// A block hash, held inline so that committing one allocates nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Key {
    len: u8,
    bytes: [u8; MAX_HASH_BYTES],
}

impl Key {
    fn from_py(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let raw = obj.cast::<PyBytes>()?.as_bytes();
        if raw.len() > MAX_HASH_BYTES {
            return Err(PyValueError::new_err(format!(
                "a block hash is at most {MAX_HASH_BYTES} bytes, got {}",
                raw.len()
            )));
        }
        let mut bytes = [0; MAX_HASH_BYTES];
        bytes[..raw.len()].copy_from_slice(raw);
        Ok(Key {
            len: u8::try_from(raw.len()).expect("checked above"),
            bytes,
        })
    }

    fn to_py<'py>(self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.bytes[..usize::from(self.len)])
    }
}

/// `handles[cache][bid]`, made the first time a block is handed out.
type Handles = Vec<Vec<Option<Py<PyLittleBlock>>>>;

/// What a pool and its block handles share.
///
/// Neither lock is held while a Python object is created: that can run the
/// garbage collector, and a finalizer reaching back into this pool would then
/// block forever on a lock its own thread holds.
struct Shared {
    pool: Mutex<JengaBlockPool<Key>>,
    handles: Mutex<Handles>,
    cache_ids: Vec<Py<PyString>>,
}

/// See [`Shared::lock`] for why poisoning is ignored.
fn lock_unpoisoned<'a, T>(py: Python<'_>, mutex: &'a Mutex<T>) -> MutexGuard<'a, T> {
    mutex
        .lock_py_attached(py)
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl Shared {
    fn lock(&self, py: Python<'_>) -> MutexGuard<'_, JengaBlockPool<Key>> {
        // A panic inside kv-core is a broken caller invariant (a double free,
        // say), which the Python pool asserted on just the same and kept
        // going after. Poisoning would turn that one error into a failure of
        // every later call, so the lock is taken regardless.
        lock_unpoisoned(py, &self.pool)
    }

    /// Returns the one handle on block `bid` of `cache`.
    fn handle(
        self: &Arc<Self>,
        py: Python<'_>,
        cache: usize,
        bid: usize,
    ) -> PyResult<Py<PyLittleBlock>> {
        if let Some(handle) = &lock_unpoisoned(py, &self.handles)[cache][bid] {
            return Ok(handle.clone_ref(py));
        }
        let handle = Py::new(
            py,
            PyLittleBlock {
                shared: Arc::downgrade(self),
                cache,
                bid,
                cache_id: self.cache_ids[cache].clone_ref(py),
            },
        )?;
        // A finalizer run by `Py::new` may have made this handle first.
        let mut handles = lock_unpoisoned(py, &self.handles);
        Ok(handles[cache][bid].get_or_insert(handle).clone_ref(py))
    }
}

/// A pool of huge blocks, each subdividable into one cache's little blocks.
///
/// See `kv_core::pool::block_pool::JengaBlockPool` for the huge/little block
/// model, the null block, and the parked/claimed states. This pool always
/// reserves the null block in huge block 0.
#[pyclass(frozen, name = "JengaBlockPool", module = "max._kv_core")]
struct PyJengaBlockPool {
    shared: Arc<Shared>,
    cache_index: HashMap<String, usize>,
    ratios: Vec<usize>,
    num_huge_blocks: usize,
    prefix_caches: Py<PyDict>,
    null_little_blocks: Py<PyDict>,
}

impl PyJengaBlockPool {
    fn cache(&self, cache_id: &str) -> PyResult<usize> {
        self.cache_index
            .get(cache_id)
            .copied()
            .ok_or_else(|| PyKeyError::new_err(cache_id.to_owned()))
    }

    /// Returns a `{cache_id: num_blocks}` mapping as one count per cache.
    fn dense(&self, demand: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
        let mut dense = vec![0; self.cache_index.len()];
        for item in demand.call_method0("items")?.try_iter()? {
            let (cache_id, num_blocks): (String, i64) = item?.extract()?;
            // No demand at all, as the Python pool read a count of 0 or less.
            dense[self.cache(&cache_id)?] = usize::try_from(num_blocks).unwrap_or(0);
        }
        Ok(dense)
    }

    /// Returns `block`'s `(cache, bid)`, refusing a block of another pool:
    /// with one pool per data-parallel replica, a mix-up would otherwise
    /// silently free or commit another replica's block.
    fn locate(&self, block: &Bound<'_, PyLittleBlock>) -> PyResult<(usize, usize)> {
        let block = block.get();
        if !std::ptr::eq(block.shared.as_ptr(), Arc::as_ptr(&self.shared)) {
            return Err(PyValueError::new_err(format!(
                "block {} of {} belongs to another pool",
                block.bid, block.cache_id
            )));
        }
        Ok((block.cache, block.bid))
    }
}

fn geometry_error(err: &JengaError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

#[pymethods]
impl PyJengaBlockPool {
    #[new]
    fn new(
        py: Python<'_>,
        num_huge_blocks: usize,
        cache_ratios: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let ratios: Vec<(String, i64)> = cache_ratios
            .call_method0("items")?
            .try_iter()?
            .map(|item| item?.extract())
            .collect::<PyResult<_>>()?;
        if let Some((cache_id, ratio)) = ratios.iter().find(|(_, ratio)| *ratio <= 0) {
            return Err(PyValueError::new_err(format!(
                "cache_ratios must be positive, found {cache_id}={ratio}"
            )));
        }

        let ratio_values: Vec<usize> = ratios
            .iter()
            .map(|&(_, ratio)| usize::try_from(ratio).expect("checked positive"))
            .collect();
        let pool = JengaBlockPool::new(
            num_huge_blocks,
            ratio_values.clone(),
            PoolOptions {
                null_block: true,
                skip_prefix_cache: false,
            },
        )
        .map_err(|err| geometry_error(&err))?;
        let handles = pool
            .little_blocks
            .iter()
            .map(|blocks| (0..blocks.len()).map(|_| None).collect())
            .collect();
        let null_bid = pool.null_bid().expect("the pool reserves one");
        let shared = Arc::new(Shared {
            pool: Mutex::new(pool),
            handles: Mutex::new(handles),
            cache_ids: ratios
                .iter()
                .map(|(cache_id, _)| PyString::new(py, cache_id).unbind())
                .collect(),
        });

        let prefix_caches = PyDict::new(py);
        let null_little_blocks = PyDict::new(py);
        for (cache, (cache_id, _)) in ratios.iter().enumerate() {
            prefix_caches.set_item(
                cache_id,
                PyPrefixCache {
                    shared: Arc::clone(&shared),
                    cache,
                },
            )?;
            null_little_blocks.set_item(cache_id, shared.handle(py, cache, null_bid)?)?;
        }

        Ok(Self {
            cache_index: ratios
                .into_iter()
                .enumerate()
                .map(|(cache, (cache_id, _))| (cache_id, cache))
                .collect(),
            shared,
            ratios: ratio_values,
            num_huge_blocks,
            prefix_caches: prefix_caches.unbind(),
            null_little_blocks: null_little_blocks.unbind(),
        })
    }

    #[getter]
    fn num_huge_blocks(&self) -> usize {
        self.num_huge_blocks
    }

    #[getter]
    fn num_allocatable_huge_blocks(&self) -> usize {
        self.num_huge_blocks - 1
    }

    #[getter]
    fn num_free_huge_blocks(&self, py: Python<'_>) -> usize {
        self.shared.lock(py).free_huge_blocks.len
    }

    #[getter]
    fn cache_ratios<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let ratios = PyDict::new(py);
        for (cache_id, ratio) in self.shared.cache_ids.iter().zip(&self.ratios) {
            ratios.set_item(cache_id, ratio)?;
        }
        Ok(ratios)
    }

    #[getter]
    fn prefix_caches(&self, py: Python<'_>) -> Py<PyDict> {
        self.prefix_caches.clone_ref(py)
    }

    #[getter]
    fn null_little_blocks(&self, py: Python<'_>) -> Py<PyDict> {
        self.null_little_blocks.clone_ref(py)
    }

    fn alloc_block(&self, py: Python<'_>, cache_id: &str) -> PyResult<Py<PyLittleBlock>> {
        let cache = self.cache(cache_id)?;
        let bid = self.shared.lock(py).alloc_block(cache).map_err(|_| {
            InsufficientBlocksError::new_err(format!("No free blocks available for {cache_id}"))
        })?;
        self.shared.handle(py, cache, bid)
    }

    fn free_block(&self, py: Python<'_>, block: &Bound<'_, PyLittleBlock>) -> PyResult<()> {
        let (cache, bid) = self.locate(block)?;
        self.shared.lock(py).free_block(cache, bid);
        Ok(())
    }

    fn touch(&self, py: Python<'_>, block: &Bound<'_, PyLittleBlock>) -> PyResult<()> {
        let (cache, bid) = self.locate(block)?;
        self.shared.lock(py).touch(cache, bid);
        Ok(())
    }

    fn uncommit_block(&self, py: Python<'_>, block: &Bound<'_, PyLittleBlock>) -> PyResult<()> {
        let (cache, bid) = self.locate(block)?;
        self.shared.lock(py).uncommit_block(cache, bid);
        Ok(())
    }

    fn commit_into_prefix_cache(
        &self,
        py: Python<'_>,
        block_hash: &Bound<'_, PyAny>,
        block: &Bound<'_, PyLittleBlock>,
    ) -> PyResult<()> {
        let (cache, bid) = self.locate(block)?;
        let key = Key::from_py(block_hash)?;
        self.shared
            .lock(py)
            .commit_into_prefix_cache(cache, key, bid);
        Ok(())
    }

    fn get_or_commit_into_prefix_cache(
        &self,
        py: Python<'_>,
        block_hash: &Bound<'_, PyAny>,
        block: &Bound<'_, PyLittleBlock>,
    ) -> PyResult<Option<Py<PyLittleBlock>>> {
        let (cache, bid) = self.locate(block)?;
        let key = Key::from_py(block_hash)?;
        let twin = self
            .shared
            .lock(py)
            .get_or_commit_into_prefix_cache(cache, key, bid);
        twin.map(|twin| self.shared.handle(py, cache, twin))
            .transpose()
    }

    fn block(&self, py: Python<'_>, cache_id: &str, bid: usize) -> PyResult<Py<PyLittleBlock>> {
        let cache = self.cache(cache_id)?;
        let ratio = self.ratios[cache];
        // Only the null block exists in huge block 0.
        if bid >= ratio * self.num_huge_blocks || (bid != 0 && bid < ratio) {
            return Err(PyIndexError::new_err(format!(
                "{cache_id} has no block {bid}"
            )));
        }
        self.shared.handle(py, cache, bid)
    }

    fn num_free_blocks(&self, py: Python<'_>, cache_id: &str) -> PyResult<usize> {
        let cache = self.cache(cache_id)?;
        Ok(self.shared.lock(py).num_free_blocks(cache))
    }

    /// Returns `(referenced, cached, free, claimed_huge_blocks,
    /// parked_huge_blocks)` for `cache_id`, as `kv_core`'s `CacheOccupancy`.
    fn occupancy(
        &self,
        py: Python<'_>,
        cache_id: &str,
    ) -> PyResult<(usize, usize, usize, usize, usize)> {
        let cache = self.cache(cache_id)?;
        let occupancy = self.shared.lock(py).occupancy(cache);
        Ok((
            occupancy.referenced,
            occupancy.cached,
            occupancy.free,
            occupancy.claimed_huge_blocks,
            occupancy.parked_huge_blocks,
        ))
    }

    #[getter]
    fn num_pristine_huge_blocks(&self, py: Python<'_>) -> usize {
        self.shared.lock(py).num_pristine_huge_blocks()
    }

    #[pyo3(signature = (demand, at_capacity = false))]
    fn can_satisfy_demand(
        &self,
        py: Python<'_>,
        demand: &Bound<'_, PyAny>,
        at_capacity: bool,
    ) -> PyResult<bool> {
        let dense = self.dense(demand)?;
        Ok(self.shared.lock(py).can_satisfy_demand(&dense, at_capacity))
    }

    #[pyo3(signature = (unit, at_capacity = false))]
    fn max_satisfiable_multiple(
        &self,
        py: Python<'_>,
        unit: &Bound<'_, PyAny>,
        at_capacity: bool,
    ) -> PyResult<usize> {
        let dense = self.dense(unit)?;
        Ok(self
            .shared
            .lock(py)
            .max_satisfiable_multiple(&dense, at_capacity))
    }

    fn reset_prefix_cache<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let purged = self.shared.lock(py).reset_prefix_cache();
        let out = PyDict::new(py);
        for (cache_id, count) in self.shared.cache_ids.iter().zip(purged) {
            out.set_item(cache_id, count)?;
        }
        Ok(out)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "JengaBlockPool(num_huge_blocks={}, cache_ratios={})",
            self.num_huge_blocks,
            self.cache_ratios(py)?.repr()?
        ))
    }
}

/// One cache's page: block `bid` of cache `cache_id`.
#[pyclass(frozen, name = "LittleKVCacheBlock", module = "max._kv_core")]
struct PyLittleBlock {
    shared: Weak<Shared>,
    cache: usize,
    bid: usize,
    cache_id: Py<PyString>,
}

impl PyLittleBlock {
    fn shared(&self) -> PyResult<Arc<Shared>> {
        self.shared
            .upgrade()
            .ok_or_else(|| PyRuntimeError::new_err("the block's pool no longer exists"))
    }
}

#[pymethods]
impl PyLittleBlock {
    #[getter]
    fn bid(&self) -> usize {
        self.bid
    }

    #[getter]
    fn cache_id(&self, py: Python<'_>) -> Py<PyString> {
        self.cache_id.clone_ref(py)
    }

    #[getter]
    fn is_null(&self) -> bool {
        self.bid == 0
    }

    #[getter]
    fn ref_cnt(&self, py: Python<'_>) -> PyResult<i64> {
        let ref_cnt = self.shared()?.lock(py).little_blocks[self.cache][self.bid].ref_cnt;
        Ok(ref_cnt)
    }

    #[getter]
    fn block_hash<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyBytes>>> {
        let key = self.shared()?.lock(py).little_blocks[self.cache][self.bid].block_hash;
        Ok(key.map(|key| key.to_py(py)))
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let block_hash = match self.block_hash(py)? {
            Some(hash) => hash.repr()?.to_string(),
            None => "None".to_owned(),
        };
        Ok(format!(
            "LittleKVCacheBlock(cache_id={}, bid={}, ref_cnt={}, block_hash={block_hash})",
            self.cache_id.bind(py).repr()?,
            self.bid,
            self.ref_cnt(py)?,
        ))
    }
}

/// One cache's committed blocks, by hash: a live, read-only view.
#[pyclass(frozen, name = "PrefixCache", module = "max._kv_core")]
struct PyPrefixCache {
    shared: Arc<Shared>,
    cache: usize,
}

impl PyPrefixCache {
    fn lookup(
        &self,
        py: Python<'_>,
        block_hash: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Py<PyLittleBlock>>> {
        // A dict answers "no" for a key of the wrong type, so a view does too.
        let Ok(key) = Key::from_py(block_hash) else {
            return Ok(None);
        };
        let bid = self.shared.lock(py).lookup_hash(self.cache, key);
        bid.map(|bid| self.shared.handle(py, self.cache, bid))
            .transpose()
    }

    fn entries(&self, py: Python<'_>) -> PyResult<Vec<(Key, Py<PyLittleBlock>)>> {
        let committed: Vec<(Key, usize)> = self.shared.lock(py).prefix_caches[self.cache]
            .iter()
            .map(|(key, &bid)| (*key, bid))
            .collect();
        committed
            .into_iter()
            .map(|(key, bid)| Ok((key, self.shared.handle(py, self.cache, bid)?)))
            .collect()
    }
}

#[pymethods]
impl PyPrefixCache {
    fn __len__(&self, py: Python<'_>) -> usize {
        self.shared.lock(py).prefix_caches[self.cache].len()
    }

    fn __contains__(&self, py: Python<'_>, block_hash: &Bound<'_, PyAny>) -> bool {
        Key::from_py(block_hash)
            .is_ok_and(|key| self.shared.lock(py).lookup_hash(self.cache, key).is_some())
    }

    fn __getitem__(
        &self,
        py: Python<'_>,
        block_hash: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyLittleBlock>> {
        self.lookup(py, block_hash)?
            .ok_or_else(|| PyKeyError::new_err(block_hash.clone().unbind()))
    }

    #[pyo3(signature = (block_hash, default = None))]
    fn get(
        &self,
        py: Python<'_>,
        block_hash: &Bound<'_, PyAny>,
        default: Option<Py<PyAny>>,
    ) -> PyResult<Option<Py<PyAny>>> {
        Ok(match self.lookup(py, block_hash)? {
            Some(block) => Some(block.into_any()),
            None => default,
        })
    }

    fn keys<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let keys: Vec<Key> = self.shared.lock(py).prefix_caches[self.cache]
            .keys()
            .copied()
            .collect();
        PyList::new(py, keys.into_iter().map(|key| key.to_py(py)))
    }

    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, self.entries(py)?.into_iter().map(|(_, block)| block))
    }

    fn items<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(
            py,
            self.entries(py)?
                .into_iter()
                .map(|(key, block)| (key.to_py(py), block)),
        )
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        // A snapshot, so committing while iterating cannot invalidate it.
        Ok(self.keys(py)?.try_iter()?.into_any())
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "PrefixCache({})",
            PyDict::from_sequence(self.items(py)?.as_any())?.repr()?
        ))
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "InsufficientBlocksError",
        m.py().get_type::<InsufficientBlocksError>(),
    )?;
    m.add_class::<PyJengaBlockPool>()?;
    m.add_class::<PyLittleBlock>()?;
    m.add_class::<PyPrefixCache>()?;
    Ok(())
}
