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
//! Block hashing, `max._kv_core.hash_block_tokens`.

use kv_core::hash::{self, BlockHashRequest, HashAlgo, TokenHashOverride};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyBytes, PyList},
};

/// Hashes the whole blocks of a token buffer into a chain of block hashes.
///
/// The stable ABI (`abi3-py310`) has no buffer protocol, so a numpy array
/// crosses as `tokens.tobytes()` plus its `itemsize`, which is one memcpy
/// where `tolist()` would build a Python int per token. `token_width` is 4 or
/// 8; wider tokens are narrowed to `int32` as `astype(np.int32)` would.
/// `overrides` holds `(index, value)` pairs, with the index relative to
/// `tokens` and the value's low 32 bits used.
///
/// The GIL is released while hashing. Raises `ValueError` for an unknown
/// `algo` or for arguments `kv_core::hash::hash_blocks` rejects.
#[pyfunction]
#[pyo3(signature = (algo, tokens, token_width, block_size, parent=None, seed=None, salt=None, overrides=Vec::new()))]
#[expect(
    clippy::too_many_arguments,
    reason = "the Python-facing signature mirrors hash_request_tokens"
)]
fn hash_block_tokens<'py>(
    py: Python<'py>,
    algo: &str,
    tokens: &Bound<'py, PyBytes>,
    token_width: usize,
    block_size: usize,
    parent: Option<&Bound<'py, PyBytes>>,
    seed: Option<&Bound<'py, PyBytes>>,
    salt: Option<&str>,
    overrides: Vec<(usize, i128)>,
) -> PyResult<Bound<'py, PyList>> {
    let algo = match algo {
        "ahash64" => HashAlgo::Ahash64,
        "sha256" => HashAlgo::Sha256,
        other => return Err(PyValueError::new_err(format!("unknown algo={other}"))),
    };
    let overrides: Vec<TokenHashOverride> = overrides
        .into_iter()
        .map(|(index, value)| TokenHashOverride {
            index,
            // The low 32 bits, as numpy's wrapping `astype(int32)` keeps them.
            value: u32::try_from(value & i128::from(u32::MAX)).unwrap_or_default(),
        })
        .collect();
    let (tokens, parent, seed) = (
        tokens.as_bytes(),
        parent.map(|p| p.as_bytes()),
        seed.map(|s| s.as_bytes()),
    );
    let hashes = py
        .detach(|| {
            hash::hash_blocks(&BlockHashRequest {
                algo,
                tokens,
                token_width,
                block_size,
                parent,
                seed,
                salt,
                overrides: &overrides,
            })
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    PyList::new(py, hashes.iter().map(|h| PyBytes::new(py, h)))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hash_block_tokens, m)?)
}
