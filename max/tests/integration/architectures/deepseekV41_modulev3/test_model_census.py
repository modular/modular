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
"""The real module's declared parameters against the checkpoint census.

Mach hands the compiled graph a bare pointer per checkpoint tensor and
runs no Python weight adapter, so at boot its weight registry checks the
MEF's declared weights against the safetensors header: every weight the
model declares must match the checkpoint tensor(s) it loads from in
dtype and byte extent. This is that check, at the source: the real
``DeepseekV41`` module is built from the full checkpoint config -- no
GPU, no weight bytes -- and every parameter it declares is resolved to
its census source(s) and compared leaf for leaf. A host transform that
writes new bytes (a dequant, an upcast) or a leaf the tree forgot
fails here in CI, not twenty minutes into an export on the engine box.
"""

from __future__ import annotations

from types import SimpleNamespace

import abi_contract
import numpy as np
import pytest
from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.architectures.deepseekV41_modulev3.deepseekV41 import (
    DeepseekV41,
)
from max.pipelines.architectures.deepseekV41_modulev3.interface import (
    STACKED_PROJECTIONS,
    V41Config,
    build_kv_params,
    to_checkpoint_name,
)
from max.pipelines.architectures.deepseekV41_modulev3.model_config import (
    DeepseekV41Config,
)
from max.pipelines.lib import KVCacheConfig
from max.tensor.sharding import DeviceMesh

_DERIVED_SUFFIX = ".ffn.global_input_scale"


def _interface_config() -> V41Config:
    full = abi_contract.hf_config("full")
    return V41Config.from_hf({"text_config": full["text_config"]})


@pytest.fixture(scope="module")
def declared() -> dict[str, tuple[DType, tuple[int, ...]]]:
    """The real module's parameters: full config, one CPU device."""
    full = abi_contract.hf_config("full")
    text = SimpleNamespace(**full["text_config"])
    interface = V41Config.from_hf({"text_config": vars(text)})
    kv = build_kv_params(
        interface, KVCacheConfig(kv_cache_page_size=128), [DeviceRef.CPU()]
    )
    config = DeepseekV41Config(
        dtype=DType.bfloat16,
        kv_params=kv,
        devices=[DeviceRef.CPU()],
        max_seq_len=256,
        **DeepseekV41Config.resolve_text_fields(text),
    )
    with F.lazy():
        module = DeepseekV41(
            config,
            kv,
            DeviceMesh((CPU(),), (1,), ("tp",)),
            text_config=text,
            token_map=np.arange(interface.vocab_size, dtype=np.int32),
        )
    return {
        name: (tensor.dtype, tuple(int(dim) for dim in tensor.shape))
        for name, tensor in module.parameters
    }


def _census_sources(name: str) -> tuple[str, ...]:
    """The checkpoint tensors ``name`` loads from, one or a stacked pair.

    A stacked leaf folds its pair's rows into one GEMM operand at load
    (bytes concatenated, never converted); every other leaf maps through
    the interface's rename table onto exactly one checkpoint tensor.
    """
    for first, second, stacked, _ in STACKED_PROJECTIONS:
        for leaf in ("weight", "scale"):
            suffix = f"{stacked}.{leaf}"
            if name.endswith(suffix):
                prefix = name[: -len(suffix)]
                return (
                    f"{prefix}{first}.{leaf}",
                    f"{prefix}{second}.{leaf}",
                )
    return (to_checkpoint_name(name),)


def _extent(dtype: DType, shape: tuple[int, ...]) -> int:
    extent = dtype.size_in_bytes
    for dim in shape:
        extent *= dim
    return extent


def test_every_declared_parameter_matches_its_census_sources(
    declared: dict[str, tuple[DType, tuple[int, ...]]],
) -> None:
    """Dtype equal and byte extent equal, per leaf, against the census.

    A stacked leaf must cover exactly its pair's bytes at the pair's
    dtype; the derived dispatch scale is the one leaf with no census
    source (the loader materializes it from the per-expert scalars it
    reduces, which the tree also holds).
    """
    census = abi_contract.census(_interface_config())
    derived = [name for name in declared if name.endswith(_DERIVED_SUFFIX)]
    assert len(derived) > 0
    for name, (dtype, shape) in declared.items():
        if name.endswith(_DERIVED_SUFFIX):
            assert dtype == DType.float32, name
            assert shape == (1,), name
            continue
        sources = _census_sources(name)
        parts = [census[source] for source in sources]
        for part_dtype, _ in parts:
            assert part_dtype == dtype, (
                f"{name}: declared {dtype}, census source is {part_dtype}"
            )
        want = 0
        for part in parts:
            extent = abi_contract.census_byte_extent(part)
            assert extent is not None, name
            want += extent
        assert _extent(dtype, shape) == want, (
            f"{name}: declared {_extent(dtype, shape)} bytes, its census"
            f" sources carry {want}"
        )


def test_every_census_tensor_is_consumed_by_a_declared_parameter(
    declared: dict[str, tuple[DType, tuple[int, ...]]],
) -> None:
    """No checkpoint tensor the tree should hold is left unconsumed.

    The complement direction: a census entry with no declared consumer
    means the tree dropped a leaf (or the adapter would load it into
    nothing), and a leaf Mach cannot feed is a boot failure the same
    way a mis-sized one is.
    """
    census = abi_contract.census(_interface_config())
    consumed = {
        source
        for name in declared
        if not name.endswith(_DERIVED_SUFFIX)
        for source in _census_sources(name)
    }
    expected = {name for name in census if not abi_contract.allow_listed(name)}
    # The trunk module this fixture builds is text-only: the router's
    # vision bias is a leaf of the vision build's leaf contract
    # (``test_param_tree`` pins that contract against the census), not
    # of the text trunk, so its census tensors stay unconsumed here.
    expected -= {
        name for name in expected if name.endswith(".ffn.gate.bias_vl")
    }
    assert not (consumed - expected), sorted(consumed - expected)[:5]
    assert not (expected - consumed), sorted(expected - consumed)[:5]
