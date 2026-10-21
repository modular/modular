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

"""Tests for the callable-composition weight adapter framework.

These tests exercise the :mod:`max.pipelines.lib.weight_loader`
combinators and walker against synthetic Module subclasses; no graph is
compiled, no weights are bound.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pytest
from max.experimental.nn import Module, ModuleList
from max.experimental.tensor import Tensor
from max.graph.weights import WeightData
from max.pipelines.lib.weight_loader import (
    HasLoaderAdapter,
    WeightLoader,
    adapt_module_loader,
    dict_loader,
    memoize,
    precompute,
    rename,
    swap_prefix,
)


def _wd(name: str, value: float = 0.0) -> WeightData:
    """Build a tiny ``WeightData`` carrying a single-element float32 array."""
    return WeightData.from_numpy(np.array([value], dtype=np.float32), name)


# ---------------------------------------------------------------------------
# Module fixtures
# ---------------------------------------------------------------------------


class _Leaf(Module[[Tensor], Tensor]):
    """Minimal Module with no adapter."""

    def forward(self, x: Tensor) -> Tensor:
        return x


class _PrefixStrip(Module[[Tensor], Tensor]):
    """Leaf whose adapter strips an outer ``prefix.`` from caller queries."""

    PREFIX: str = "renamed."

    @staticmethod
    def adapt_loader(loader: WeightLoader) -> WeightLoader:
        # Caller asks for "renamed.X"; query the source for "X".
        return rename(
            loader,
            lambda name: name.removeprefix(_PrefixStrip.PREFIX),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x


# ---------------------------------------------------------------------------
# Protocol membership
# ---------------------------------------------------------------------------


class TestProtocols:
    def test_dict_loader_satisfies_weight_loader(self) -> None:
        loader = dict_loader({"a": _wd("a")})
        assert isinstance(loader, WeightLoader)

    def test_leaf_with_adapter_is_recognized(self) -> None:
        assert isinstance(_PrefixStrip(), HasLoaderAdapter)

    def test_leaf_without_adapter_is_not(self) -> None:
        assert not isinstance(_Leaf(), HasLoaderAdapter)


# ---------------------------------------------------------------------------
# dict_loader
# ---------------------------------------------------------------------------


class TestDictLoader:
    def test_resolves_by_name(self) -> None:
        wd = _wd("foo")
        loader = dict_loader({"foo": wd})
        assert loader("foo") is wd

    def test_keys_filters_by_prefix(self) -> None:
        loader = dict_loader(
            {"a.x": _wd("a.x"), "a.y": _wd("a.y"), "b.z": _wd("b.z")}
        )
        assert set(loader.keys("a.")) == {"a.x", "a.y"}
        assert set(loader.keys()) == {"a.x", "a.y", "b.z"}

    def test_missing_key_raises(self) -> None:
        loader = dict_loader({"a": _wd("a")})
        with pytest.raises(KeyError):
            loader("missing")


# ---------------------------------------------------------------------------
# rename combinator
# ---------------------------------------------------------------------------


class TestRename:
    def test_translates_queries(self) -> None:
        base = dict_loader({"source.weight": _wd("v")})
        renamed = rename(
            base, lambda name: name.replace("canonical.", "source.")
        )
        assert renamed("canonical.weight") is base("source.weight")

    def test_forwards_keys_unchanged_without_inverse(self) -> None:
        # Without ``key_inv``, ``rename`` is forward-only: keys() reflects
        # the inner namespace, not the renamed outward view.
        base = dict_loader({"src.a": _wd("a")})
        renamed = rename(base, lambda n: n.removeprefix("dst."))
        assert set(renamed.keys()) == {"src.a"}

    def test_key_inv_exposes_outward_namespace(self) -> None:
        base = dict_loader({"src.a": _wd("a"), "src.b": _wd("b")})
        renamed = rename(
            base,
            fn=lambda n: n.replace("dst.", "src.", 1),
            key_inv=lambda n: n.replace("src.", "dst.", 1),
        )
        assert set(renamed.keys()) == {"dst.a", "dst.b"}
        # Prefix filter applies to the translated (outward) name.
        assert set(renamed.keys("dst.")) == {"dst.a", "dst.b"}

    def test_missing_target_raises(self) -> None:
        base = dict_loader({"a": _wd("a")})
        renamed = rename(base, lambda n: n + ".missing")
        with pytest.raises(KeyError):
            renamed("a")


class TestSwapPrefix:
    def test_query_and_keys_translate_bidirectionally(self) -> None:
        base = dict_loader(
            {"transformer.attn.weight": _wd("w"), "other.x": _wd("x")}
        )
        swapped = swap_prefix(base, outer="denoiser", inner="transformer")
        # Outward queries route to the inner prefix.
        assert swapped("denoiser.attn.weight") is base(
            "transformer.attn.weight"
        )
        # Untouched keys pass through.
        assert swapped("other.x") is base("other.x")
        # ``keys`` reflects the outward view.
        assert set(swapped.keys()) == {"denoiser.attn.weight", "other.x"}
        assert set(swapped.keys("denoiser.")) == {"denoiser.attn.weight"}


# ---------------------------------------------------------------------------
# precompute combinator
# ---------------------------------------------------------------------------


class TestPrecompute:
    def test_consumed_keys_replaced_by_transform(self) -> None:
        base = dict_loader(
            {
                "qkv.weight": _wd("qkv"),
                "norm.weight": _wd("norm"),
            }
        )

        def split(d: dict[str, WeightData]) -> dict[str, WeightData]:
            qkv = d["qkv.weight"]
            return {
                "q.weight": qkv,
                "k.weight": qkv,
                "v.weight": qkv,
            }

        loader = precompute(base, ["qkv.weight"], split)
        assert loader("q.weight") is base("qkv.weight")
        assert loader("k.weight") is base("qkv.weight")
        # Unrelated keys still pass through.
        assert loader("norm.weight") is base("norm.weight")

    def test_consumed_key_no_longer_reachable(self) -> None:
        base = dict_loader({"qkv.weight": _wd("qkv")})
        loader = precompute(
            base,
            ["qkv.weight"],
            lambda d: {"q.weight": d["qkv.weight"]},
        )
        with pytest.raises(KeyError, match="consumed by a precompute"):
            loader("qkv.weight")

    def test_keys_reflects_transformed_namespace(self) -> None:
        base = dict_loader(
            {"qkv.weight": _wd("qkv"), "norm.weight": _wd("norm")}
        )
        loader = precompute(
            base,
            ["qkv.weight"],
            lambda d: {
                "q.weight": d["qkv.weight"],
                "k.weight": d["qkv.weight"],
                "v.weight": d["qkv.weight"],
            },
        )
        # qkv.weight is gone; q/k/v.weight are exposed; norm still there.
        assert set(loader.keys()) == {
            "q.weight",
            "k.weight",
            "v.weight",
            "norm.weight",
        }

    def test_transform_can_drop_keys(self) -> None:
        base = dict_loader({"discard.weight": _wd("d")})
        loader = precompute(base, ["discard.weight"], lambda d: {})
        with pytest.raises(KeyError):
            loader("discard.weight")
        assert set(loader.keys()) == set()

    def test_empty_keys_passes_through(self) -> None:
        base = dict_loader({"a": _wd("a")})
        loader = precompute(base, [], lambda d: {})
        assert loader("a") is base("a")


# ---------------------------------------------------------------------------
# memoize combinator
# ---------------------------------------------------------------------------


class _CountingLoader:
    """Test-only loader that counts ``__call__`` invocations."""

    def __init__(self, state: dict[str, WeightData]) -> None:
        self._state = state
        self.call_count = 0

    def __call__(self, name: str) -> WeightData:
        self.call_count += 1
        return self._state[name]

    def keys(self, prefix: str = "") -> Iterable[str]:
        return (k for k in self._state if k.startswith(prefix))


class TestMemoize:
    def test_caches_repeat_queries(self) -> None:
        base = _CountingLoader({"a": _wd("a")})
        loader = memoize(base)
        first = loader("a")
        second = loader("a")
        assert first is second
        assert base.call_count == 1

    def test_forwards_keys(self) -> None:
        base = _CountingLoader({"a": _wd("a"), "b": _wd("b")})
        loader = memoize(base)
        assert set(loader.keys()) == {"a", "b"}


# ---------------------------------------------------------------------------
# Walker semantics
# ---------------------------------------------------------------------------


class TestAdaptModuleLoader:
    def test_no_adapters_returns_base_passthrough(self) -> None:
        base = dict_loader({"a": _wd("a")})
        out = adapt_module_loader(_Leaf(), base)
        assert out("a") is base("a")

    def test_root_adapter_wraps_whole_namespace(self) -> None:
        # Root adapter strips "renamed." from every query.
        base = dict_loader({"foo": _wd("foo")})
        out = adapt_module_loader(_PrefixStrip(), base)
        assert out("renamed.foo") is base("foo")

    def test_child_adapter_sees_only_its_scope(self) -> None:
        class Composite(Module[[Tensor], Tensor]):
            def __init__(self) -> None:
                self.encoder = _PrefixStrip()

            def forward(self, x: Tensor) -> Tensor:
                return self.encoder(x)

        base = dict_loader({"encoder.foo": _wd("foo")})
        out = adapt_module_loader(Composite(), base)
        # Caller asks for the "renamed.foo" namespace below "encoder.";
        # encoder strips "renamed." and queries the inner loader with
        # "foo", which _scoped re-prefixes back to "encoder.foo".
        assert out("encoder.renamed.foo") is base("encoder.foo")

    def test_sibling_adapters_independent(self) -> None:
        class Flux(Module[[Tensor], Tensor]):
            def __init__(self) -> None:
                self.encoder = _PrefixStrip()
                self.decoder = _PrefixStrip()

            def forward(self, x: Tensor) -> Tensor:
                return x

        base = dict_loader(
            {
                "encoder.a": _wd("ea"),
                "decoder.b": _wd("db"),
                "stray.c": _wd("sc"),
            }
        )
        out = adapt_module_loader(Flux(), base)
        assert out("encoder.renamed.a") is base("encoder.a")
        assert out("decoder.renamed.b") is base("decoder.b")
        # Untouched siblings pass through.
        assert out("stray.c") is base("stray.c")

    def test_grandchildren_descended_into(self) -> None:
        class Middle(Module[[Tensor], Tensor]):
            def __init__(self) -> None:
                self.deep = _PrefixStrip()

            def forward(self, x: Tensor) -> Tensor:
                return x

        class Root(Module[[Tensor], Tensor]):
            def __init__(self) -> None:
                self.middle = Middle()

            def forward(self, x: Tensor) -> Tensor:
                return x

        base = dict_loader({"middle.deep.foo": _wd("foo")})
        out = adapt_module_loader(Root(), base)
        assert out("middle.deep.renamed.foo") is base("middle.deep.foo")


# ---------------------------------------------------------------------------
# ModuleList traversal
# ---------------------------------------------------------------------------


class TestModuleListTraversal:
    def test_module_list_children_each_run(self) -> None:
        class Encoder(Module[[Tensor], Tensor]):
            def __init__(self) -> None:
                self.layers = ModuleList([_PrefixStrip(), _PrefixStrip()])

            def forward(self, x: Tensor) -> Tensor:
                return x

        base = dict_loader({"layers.0.a": _wd("0.a"), "layers.1.b": _wd("1.b")})
        out = adapt_module_loader(Encoder(), base)
        assert out("layers.0.renamed.a") is base("layers.0.a")
        assert out("layers.1.renamed.b") is base("layers.1.b")

    def test_module_list_root(self) -> None:
        base = dict_loader({"0.a": _wd("0.a"), "1.b": _wd("1.b")})
        root = ModuleList([_PrefixStrip(), _PrefixStrip()])
        out = adapt_module_loader(root, base)
        assert out("0.renamed.a") is base("0.a")
        assert out("1.renamed.b") is base("1.b")


# ---------------------------------------------------------------------------
# Raw container guards
# ---------------------------------------------------------------------------


class TestRawContainerGuard:
    def test_raw_list_of_modules_raises(self) -> None:
        class Bad(Module[[Tensor], Tensor]):
            def __init__(self) -> None:
                self.layers = [_PrefixStrip()]  # should be ModuleList

            def forward(self, x: Tensor) -> Tensor:
                return x

        with pytest.raises(ValueError, match=r"layers\[0\].*ModuleList"):
            adapt_module_loader(Bad(), dict_loader({}))

    def test_raw_dict_of_modules_raises(self) -> None:
        class Bad(Module[[Tensor], Tensor]):
            def __init__(self) -> None:
                self.by_name = {"a": _PrefixStrip()}

            def forward(self, x: Tensor) -> Tensor:
                return x

        with pytest.raises(ValueError, match=r"by_name\['a'\].*ModuleList"):
            adapt_module_loader(Bad(), dict_loader({}))

    def test_plain_data_lists_pass_through(self) -> None:
        class Encoder(Module[[Tensor], Tensor]):
            def __init__(self) -> None:
                self.shape = [1, 2, 3]
                self.layer = _PrefixStrip()

            def forward(self, x: Tensor) -> Tensor:
                return x

        base = dict_loader({"layer.foo": _wd("foo")})
        out = adapt_module_loader(Encoder(), base)
        assert out("layer.renamed.foo") is base("layer.foo")

    def test_guard_fires_on_nested_module(self) -> None:
        class Inner(Module[[Tensor], Tensor]):
            def __init__(self) -> None:
                self.layers = [_PrefixStrip()]

            def forward(self, x: Tensor) -> Tensor:
                return x

        class Root(Module[[Tensor], Tensor]):
            def __init__(self) -> None:
                self.inner = Inner()

            def forward(self, x: Tensor) -> Tensor:
                return x

        with pytest.raises(ValueError, match=r"inner\.layers\[0\]"):
            adapt_module_loader(Root(), dict_loader({}))


# ---------------------------------------------------------------------------
# End-to-end FLUX-shaped example
# ---------------------------------------------------------------------------


class TestFluxLikeExample:
    """End-to-end demonstration of the loader-composition pattern.

    Mirrors the FLUX.2 manifest shape: a flat role-prefixed source dict
    (``text_encoder.*``, ``transformer.*``) goes in; the Module tree's
    adapters cooperate to expose canonical Module-tree names
    (``encoder.layers.*``, ``denoiser.attn.q_proj.weight``, ...).
    """

    def test_text_encoder_prefix_strip_and_denoiser_qkv_split(self) -> None:
        # --- TextEncoder: HF-style prefix strip ---
        class TextEncoder(Module[[Tensor], Tensor]):
            @staticmethod
            def adapt_loader(loader: WeightLoader) -> WeightLoader:
                # Module asks for "layers.0.input_layernorm.weight" etc.
                # Source has these under "language_model.model.layers.0...".
                # Fall through several source-prefix variants.
                def to_source(name: str) -> str:
                    for src in (
                        "language_model.model.",
                        "model.",
                        "",
                    ):
                        candidate = f"{src}{name}"
                        if any(k == candidate for k in loader.keys(src)):
                            return candidate
                    raise KeyError(name)

                return rename(loader, to_source)

            def forward(self, x: Tensor) -> Tensor:
                return x

        # --- Denoiser: bulk QKV split via precompute ---
        def _split_qkv(
            d: dict[str, WeightData],
        ) -> dict[str, WeightData]:
            # Demo splitter: emit q/k/v keys pointing at the same fused
            # WeightData (the real implementation slices the tensor).
            out: dict[str, WeightData] = {}
            for key, value in d.items():
                base_path = key.replace(".attn.qkv_proj.", ".attn.")
                for proj in ("q_proj", "k_proj", "v_proj"):
                    out[base_path.replace(".attn.", f".attn.{proj}.")] = value
            return out

        class Denoiser(Module[[Tensor], Tensor]):
            @staticmethod
            def adapt_loader(loader: WeightLoader) -> WeightLoader:
                # Find fused QKV weights in the source namespace and
                # split them eagerly. Everything else stays lazy.
                qkv_keys = [
                    k
                    for k in loader.keys()  # noqa: SIM118 (WeightLoader.keys)
                    if ".attn.qkv_proj." in k
                ]
                if qkv_keys:
                    loader = precompute(loader, qkv_keys, _split_qkv)
                return loader

            def forward(self, x: Tensor) -> Tensor:
                return x

        # --- Composite root, mirroring ``FLUXModule`` ---
        class FluxModule(Module[[Tensor], Tensor]):
            @staticmethod
            def adapt_loader(loader: WeightLoader) -> WeightLoader:
                # Caller's Module attribute is "denoiser" but the
                # manifest role is "transformer". Bijective prefix swap
                # so that ``loader.keys()`` inside the denoiser's
                # adapter reflects the canonical (outward) namespace.
                return swap_prefix(
                    loader, outer="denoiser", inner="transformer"
                )

            def __init__(self) -> None:
                self.text_encoder = TextEncoder()
                self.denoiser = Denoiser()

            def forward(self, x: Tensor) -> Tensor:
                return x

        # Source: role-prefixed flat dict as produced by
        # ``ModelManifest.loader()``'s underlying state assembly.
        qkv_wd = _wd("qkv")
        manifest_state = {
            "text_encoder.language_model.model.layers.0.input_layernorm.weight": _wd(
                "te0"
            ),
            "text_encoder.language_model.model.layers.1.input_layernorm.weight": _wd(
                "te1"
            ),
            "transformer.blocks.0.attn.qkv_proj.weight": qkv_wd,
            "transformer.blocks.0.norm.weight": _wd("n0"),
        }

        loader = adapt_module_loader(FluxModule(), dict_loader(manifest_state))

        # TextEncoder canonical -> source via HF-style prefix strip.
        assert (
            loader("text_encoder.layers.0.input_layernorm.weight")
            is manifest_state[
                "text_encoder.language_model.model.layers.0.input_layernorm.weight"
            ]
        )

        # Denoiser canonical -> source via "denoiser." -> "transformer."
        # rebase + QKV split.
        assert loader("denoiser.blocks.0.attn.q_proj.weight") is qkv_wd
        assert loader("denoiser.blocks.0.attn.k_proj.weight") is qkv_wd
        assert loader("denoiser.blocks.0.attn.v_proj.weight") is qkv_wd

        # Untouched denoiser keys still resolve through the rebase
        # without going through precompute.
        assert (
            loader("denoiser.blocks.0.norm.weight")
            is manifest_state["transformer.blocks.0.norm.weight"]
        )
