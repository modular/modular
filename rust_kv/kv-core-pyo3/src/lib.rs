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
//! Python bindings for `kv-core`, shipped in the `max` wheel as `max._kv_core`.
//!
//! The binding source stays closed; the public surface is the hand-written
//! stub `max/python/max/_kv_core.pyi`, which must change with this file.
//!
//! The Jenga block pool is in [`pool`] and block hashing in [`hash`].
//!
//! A leaf's shape crosses the boundary as a [`PyLeafShape`] rather than as
//! MAX's `KVCacheGroupId`, so the Rust side needs no MAX type and MAX keeps the
//! window arithmetic (`KVCacheGroupId.blocks_in_window`) that builds it.

mod hash;
mod pool;

use kv_core::prefix_hit::{self, LeafShape};
use pyo3::{exceptions::PyValueError, prelude::*};

/// Reads a residency mask from a Python sequence of bools or a 1-D numpy
/// array.
///
/// A numpy array is not a `collections.abc.Sequence`, so it goes through
/// `tolist()` first. That costs one Python object per block: the stable ABI
/// (`abi3-py310`) has no buffer protocol before 3.11, which is what would let
/// a bool array be read in place.
fn residency(mask: &Bound<'_, PyAny>) -> PyResult<Vec<bool>> {
    if mask.hasattr("tolist")? {
        return mask.call_method0("tolist")?.extract();
    }
    mask.extract()
}

/// A leaf's attention shape, which selects the rule that reads its residency.
#[pyclass(
    frozen,
    eq,
    hash,
    from_py_object,
    name = "LeafShape",
    module = "max._kv_core"
)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct PyLeafShape(LeafShape);

#[pymethods]
impl PyLeafShape {
    /// A leaf that reads its whole history.
    #[staticmethod]
    fn full() -> Self {
        Self(LeafShape::Full)
    }

    /// A leaf that reads the last `blocks_in_window` blocks.
    ///
    /// Raises `ValueError` for a window of zero blocks: `kv-core` only
    /// debug-asserts it, which would surface as a `PanicException` that no
    /// `except Exception` catches.
    #[staticmethod]
    fn sliding_window(blocks_in_window: usize) -> PyResult<Self> {
        if blocks_in_window == 0 {
            return Err(PyValueError::new_err(
                "a sliding-window leaf spans at least one block; got blocks_in_window=0",
            ));
        }
        Ok(Self(LeafShape::SlidingWindow { blocks_in_window }))
    }

    /// A leaf that holds one state per published boundary.
    #[staticmethod]
    fn recurrent() -> Self {
        Self(LeafShape::Recurrent)
    }

    /// A leaf that is never published and so never narrows a hit.
    #[staticmethod]
    fn scratch() -> Self {
        Self(LeafShape::Scratch)
    }

    /// How much of `resident` a leaf of this shape can reuse, counted from
    /// the start.
    fn longest_hit(&self, resident: &Bound<'_, PyAny>) -> PyResult<usize> {
        let mask = residency(resident)?;
        Ok(self.0.longest_hit(mask.len(), |i| mask[i]))
    }

    fn __repr__(&self) -> String {
        match self.0 {
            LeafShape::Full => "LeafShape.full()".to_owned(),
            LeafShape::SlidingWindow { blocks_in_window } => {
                format!("LeafShape.sliding_window({blocks_in_window})")
            }
            LeafShape::Recurrent => "LeafShape.recurrent()".to_owned(),
            LeafShape::Scratch => "LeafShape.scratch()".to_owned(),
        }
    }
}

/// The longest prefix every leaf can serve at once.
///
/// `leaves` holds one `(shape, resident)` pair per leaf. Leaves routinely share
/// a shape, so this is a sequence rather than a mapping. Returns `0` when there
/// are no leaves or they cannot agree on a prefix.
///
/// Raises `ValueError` if the masks are not all the same width: they index one
/// chain of hashes, so a short one would quietly answer about a different
/// prefix than the rest.
#[pyfunction]
fn longest_joint_prefix_hit(leaves: Vec<(PyLeafShape, Bound<'_, PyAny>)>) -> PyResult<usize> {
    let mut shapes = Vec::with_capacity(leaves.len());
    let mut masks = Vec::with_capacity(leaves.len());
    for (shape, mask) in leaves {
        shapes.push(shape.0);
        masks.push(residency(&mask)?);
    }

    let Some(width) = masks.first().map(Vec::len) else {
        return Ok(0);
    };
    if masks.iter().any(|mask| mask.len() != width) {
        let mut widths: Vec<usize> = masks.iter().map(Vec::len).collect();
        widths.sort_unstable();
        widths.dedup();
        return Err(PyValueError::new_err(format!(
            "every leaf's mask covers the same chain; got widths {widths:?}"
        )));
    }

    Ok(prefix_hit::longest_joint_prefix_hit(
        width,
        masks.len(),
        |leaf, candidate| shapes[leaf].longest_hit(candidate, |i| masks[leaf][i]),
    ))
}

/// How many blocks a leaf actually holds of a hit `num_hit_blocks` deep.
#[pyfunction]
fn blocks_held_of_hit(num_hit_blocks: usize, blocks_in_window: Option<usize>) -> usize {
    prefix_hit::blocks_held_of_hit(num_hit_blocks, blocks_in_window)
}

/// Host-side KV cache bookkeeping shared by MAX and Mach.
// CPython looks up `PyInit__kv_core` in the cdylib at import.
#[pymodule]
fn _kv_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLeafShape>()?;
    m.add_function(wrap_pyfunction!(longest_joint_prefix_hit, m)?)?;
    m.add_function(wrap_pyfunction!(blocks_held_of_hit, m)?)?;
    hash::register(m)?;
    pool::register(m)?;
    Ok(())
}
