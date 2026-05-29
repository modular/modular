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

"""Callable-composition weight adapter framework for ``Module``-trees.

Leaves the source state dict untouched and translates the *queries* a
``Module`` makes against it. The
:meth:`~max.pipelines.lib.model_manifest.ModelManifest.loader` entry
point produces the base loader; per-Module
:class:`HasLoaderAdapter` implementations wrap it as the tree is
walked, so each sub-Module sees weights in its own canonical
namespace.

The shapes:

* :class:`WeightLoader` -- a ``Callable[[str], WeightData]`` augmented
  with a ``keys(prefix)`` enumerator so adapters can scan the namespace
  without materialising values.
* :data:`WeightAdapter` -- ``Callable[[WeightLoader], WeightLoader]``;
  wraps a source loader to serve queries in a Module's canonical
  namespace.
* :class:`HasLoaderAdapter` -- ``runtime_checkable`` Protocol marking a
  ``Module`` that declares a ``adapt_loader(loader) -> loader``.

The framework provides four combinators -- :func:`rename`,
:func:`swap_prefix`, :func:`precompute`, :func:`memoize` -- so adapters
can pick lazy per-key translation or eager-within-a-group
precomputation as the transform requires. Bulk transforms
(stacked-QKV split, NVFP4 weight + scale pairing) stay
``dict -> dict`` functions wrapped via :func:`precompute`; bijective
per-key renames go through :func:`swap_prefix` (which keeps ``keys``
faithful to the outward namespace); irreversible per-key renames go
through :func:`rename`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Protocol, runtime_checkable

from max.experimental.nn import Module
from max.graph.weights import WeightData, Weights


@runtime_checkable
class WeightLoader(Protocol):
    """A weight source. Resolves canonical names and enumerates its namespace.

    Resolving a name returns a :class:`~max.graph.weights.WeightData`;
    enumeration returns the names that resolve. Both operations are
    expected to be cheap relative to materialising tensor bytes, so a
    safetensors-backed loader can lazily mmap on resolve and serve
    ``keys`` from the index.
    """

    def __call__(self, name: str) -> WeightData:
        """Resolves ``name`` to a :class:`~max.graph.weights.WeightData`."""
        ...

    def keys(self, prefix: str = "") -> Iterable[str]:
        """Enumerates names in this loader's namespace starting with ``prefix``."""
        ...


WeightAdapter = Callable[[WeightLoader], WeightLoader]
"""Wraps a ``WeightLoader`` to translate queries for one Module's params.

An adapter receives an inner loader serving the namespace below it in
the Module tree and returns a loader that serves the same namespace as
the Module's caller expects to query it.
"""


@runtime_checkable
class HasLoaderAdapter(Protocol):
    """A ``Module`` that declares how to translate queries against a source.

    The renamer is intentionally a static method: query translation is
    pure name manipulation (plus optional bulk precomputation) and
    should not depend on Module instance state.
    """

    @staticmethod
    def adapt_loader(loader: WeightLoader) -> WeightLoader:
        """Returns a wrapped loader serving this Module's canonical names."""
        ...


# ---------------------------------------------------------------------------
# Combinators
# ---------------------------------------------------------------------------


class _LoaderImpl:
    """Plain ``WeightLoader`` from two callables. Used by the combinators."""

    def __init__(
        self,
        call: Callable[[str], WeightData],
        keys_fn: Callable[[str], Iterable[str]],
    ) -> None:
        self._call = call
        self._keys = keys_fn

    def __call__(self, name: str) -> WeightData:
        return self._call(name)

    def keys(self, prefix: str = "") -> Iterable[str]:
        return self._keys(prefix)


def rename(
    loader: WeightLoader,
    fn: Callable[[str], str],
    *,
    key_inv: Callable[[str], str] | None = None,
) -> WeightLoader:
    """Wraps ``loader`` so each query name is translated through ``fn``.

    Lazy: ``fn`` is applied only when a query arrives.

    By default ``keys`` is forwarded to the inner loader unchanged --
    the wrapped loader exposes the inner namespace, not a
    reverse-translated view, because ``fn`` is not required to be
    invertible. When ``key_inv`` is supplied it is applied to each inner
    key, so ``keys`` reflects the outward namespace; this is the right
    choice for prefix swaps and other bijective renames (see
    :func:`swap_prefix`).

    Args:
        loader: Inner loader to query after translation.
        fn: Outgoing-query rewriter. Receives the name the caller asks
            for; returns the name to query against ``loader``.
        key_inv: Optional inward-key-to-outward-key translator. When
            provided, ``keys`` yields ``key_inv(k)`` for every inner key
            ``k``; the ``prefix`` filter is applied to the translated
            name.

    Returns:
        A new :class:`WeightLoader` that forwards every call through
        ``fn`` and, when ``key_inv`` is provided, exposes the outward
        namespace via ``keys``.
    """
    if key_inv is None:
        return _LoaderImpl(
            call=lambda name: loader(fn(name)),
            keys_fn=lambda p: loader.keys(p),
        )

    def keys_fn(p: str = "") -> Iterable[str]:
        for k in loader.keys():  # noqa: SIM118 (WeightLoader.keys, not dict)
            translated = key_inv(k)
            if translated.startswith(p):
                yield translated

    return _LoaderImpl(
        call=lambda name: loader(fn(name)),
        keys_fn=keys_fn,
    )


def swap_prefix(loader: WeightLoader, outer: str, inner: str) -> WeightLoader:
    """Bijective prefix swap: caller sees ``outer.*``, source has ``inner.*``.

    Convenience wrapper around :func:`rename` that builds both the
    forward (query) and inverse (keys) translators for a simple prefix
    swap. Use this when a Module attribute's name does not match the
    manifest role it draws from (for example, ``FLUXModule.denoiser``
    pulling from the ``transformer`` manifest role).

    Args:
        loader: Inner loader serving the ``inner.*`` namespace.
        outer: Caller-facing prefix (without trailing dot).
        inner: Source-facing prefix (without trailing dot).

    Returns:
        A new :class:`WeightLoader` where querying ``outer.X`` resolves
        to ``loader("inner.X")``, and ``keys`` yields ``outer.X`` for
        each inner key ``inner.X``.
    """
    outer_dot = f"{outer.rstrip('.')}."
    inner_dot = f"{inner.rstrip('.')}."

    def fn(name: str) -> str:
        if name.startswith(outer_dot):
            return f"{inner_dot}{name[len(outer_dot) :]}"
        return name

    def key_inv(name: str) -> str:
        if name.startswith(inner_dot):
            return f"{outer_dot}{name[len(inner_dot) :]}"
        return name

    return rename(loader, fn, key_inv=key_inv)


def precompute(
    loader: WeightLoader,
    keys: Iterable[str],
    transform: Callable[[dict[str, WeightData]], dict[str, WeightData]],
) -> WeightLoader:
    """Eagerly pulls ``keys`` from ``loader``, runs them through ``transform``.

    The returned loader serves the ``transform`` output and falls
    through to ``loader`` for everything else. Source keys that were
    consumed by ``transform`` are no longer reachable on the returned
    loader -- they have been replaced by whatever ``transform``
    produced.

    Eager-within-a-group: only the listed ``keys`` are materialised
    here; every other name stays lazy. This is the right hook for
    transforms whose input and output namespaces don't line up
    one-to-one (stacked-QKV split, NVFP4 weight + scale pairing,
    fused-tensor unpacking).

    Args:
        loader: Source loader to draw the group from.
        keys: Names to consume from ``loader``. Each is queried once.
        transform: Pure function from the consumed sub-dict to the
            replacement sub-dict. May emit any names it wants; emitting
            none is equivalent to dropping the group.

    Returns:
        A new :class:`WeightLoader` serving ``transform``'s output and
        falling through to ``loader`` for any name not in ``keys``.
    """
    consumed = set(keys)
    source = {k: loader(k) for k in consumed}
    transformed = transform(source)

    def call(name: str) -> WeightData:
        if name in transformed:
            return transformed[name]
        if name in consumed:
            raise KeyError(
                f"`{name}` was consumed by a precompute group and "
                "replaced by the transform output. The original key "
                "is no longer reachable."
            )
        return loader(name)

    def keys_fn(prefix: str = "") -> Iterable[str]:
        for k in loader.keys(prefix):
            if k in consumed or k in transformed:
                continue
            yield k
        for k in transformed:
            if k.startswith(prefix):
                yield k

    return _LoaderImpl(call=call, keys_fn=keys_fn)


def memoize(loader: WeightLoader) -> WeightLoader:
    """Caches resolved ``WeightData`` so repeat queries are free.

    Useful as the outermost wrapper when a downstream consumer asks for
    the same name multiple times (for example, validation passes after
    the compile-time bind).

    ``keys`` is forwarded to the inner loader unchanged.

    Args:
        loader: Inner loader to memoise.

    Returns:
        A new :class:`WeightLoader` that caches every successful
        resolution.
    """
    cache: dict[str, WeightData] = {}

    def call(name: str) -> WeightData:
        if name not in cache:
            cache[name] = loader(name)
        return cache[name]

    return _LoaderImpl(call=call, keys_fn=lambda p: loader.keys(p))


# ---------------------------------------------------------------------------
# Tree walker
# ---------------------------------------------------------------------------


def _check_no_unwrapped_module_containers(
    module: Module[..., Any], path: str
) -> None:
    """Raises if any list/tuple/dict attribute holds Modules.

    Homogeneous collections of sub-Modules must be wrapped in
    :class:`max.experimental.nn.ModuleList` so the walker visits them
    under stringified-index dotted paths. A raw container of Modules
    bypasses :meth:`Module.descendants` and would silently drop its
    adapters.
    """
    for name, value in vars(module).items():
        if isinstance(value, Module):
            continue
        if isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                if isinstance(item, Module):
                    where = f"{path}.{name}[{i}]" if path else f"{name}[{i}]"
                    raise ValueError(
                        f"{where}: ``Module`` instances inside a raw "
                        f"``{type(value).__name__}`` attribute are not "
                        "visited by ``adapt_module_loader`` and their "
                        "adapters would be silently skipped. Wrap them "
                        "in ``max.experimental.nn.ModuleList`` instead."
                    )
        elif isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, Module):
                    where = (
                        f"{path}.{name}[{key!r}]"
                        if path
                        else f"{name}[{key!r}]"
                    )
                    raise ValueError(
                        f"{where}: ``Module`` instances inside a raw "
                        "``dict`` attribute are not visited by "
                        "``adapt_module_loader`` and their adapters "
                        "would be silently skipped. Wrap them in "
                        "``max.experimental.nn.ModuleList`` (keyed by "
                        "stringified index) instead."
                    )


def _scoped(prefix: str, adapter: WeightAdapter) -> WeightAdapter:
    """Restricts an un-prefixed adapter to the ``prefix.*`` query namespace.

    Returns an adapter whose ``__call__`` routes queries beginning with
    ``f"{prefix}."`` into the wrapped one (with the prefix stripped),
    and passes every other query straight through to the inner loader.

    The wrapped adapter sees a loader whose own namespace is relative
    to ``prefix`` -- queries get the prefix re-added before reaching
    the original ``inner``.
    """
    if not prefix:
        return adapter
    dotted = f"{prefix}."

    def wrap(inner: WeightLoader) -> WeightLoader:
        unprefixed_inner = _LoaderImpl(
            call=lambda name: inner(f"{dotted}{name}"),
            keys_fn=lambda p: (
                k[len(dotted) :]
                for k in inner.keys(f"{dotted}{p}")
                if k.startswith(dotted)
            ),
        )
        adapted = adapter(unprefixed_inner)

        def call(name: str) -> WeightData:
            if name.startswith(dotted):
                return adapted(name[len(dotted) :])
            return inner(name)

        def keys_fn(p: str = "") -> Iterable[str]:
            # Outer view: everything inner exposes that isn't under
            # ``prefix.`` (untouched), plus the adapted namespace
            # re-prefixed under ``prefix.``.
            for k in inner.keys(p):
                if not k.startswith(dotted):
                    yield k
            if p.startswith(dotted):
                sub = p[len(dotted) :]
            elif dotted.startswith(p):
                sub = ""
            else:
                return
            for k in adapted.keys(sub):
                yield f"{dotted}{k}"

        return _LoaderImpl(call=call, keys_fn=keys_fn)

    return wrap


def adapt_module_loader(
    module: Module[..., Any],
    base: WeightLoader,
) -> WeightLoader:
    """Walks ``module``'s tree and composes every :class:`HasLoaderAdapter`.

    For each Module in the tree (root first, then each descendant in
    top-down order) that implements :class:`HasLoaderAdapter`, wraps
    the current loader with that Module's ``adapt_loader``, scoped to
    the Module's path within the tree. Modules that do not implement
    :class:`HasLoaderAdapter` are skipped (the walker still descends
    into them via :meth:`Module.descendants`).

    The walk uses :attr:`Module.descendants`. Homogeneous collections
    of sub-Modules must be wrapped in
    :class:`max.experimental.nn.ModuleList` (or a subclass such as
    :class:`~max.experimental.nn.Sequential`) so they are visited with
    stringified-index names. A raw ``list``, ``tuple``, or ``dict``
    attribute whose elements are ``Module`` instances raises
    ``ValueError`` to surface the silent skip eagerly.

    Args:
        module: Root Module whose adapter chain to assemble.
        base: Source loader to wrap.

    Returns:
        A new :class:`WeightLoader` that satisfies every adapter in the
        tree when queried with canonical Module-tree parameter names.

    Raises:
        ValueError: If any descendant exposes ``Module`` instances
            inside a raw ``list``, ``tuple``, or ``dict`` attribute.
    """
    loader = base
    _check_no_unwrapped_module_containers(module, path="")
    if isinstance(module, HasLoaderAdapter):
        loader = module.adapt_loader(loader)
    for path, child in module.descendants:
        _check_no_unwrapped_module_containers(child, path=path)
        if isinstance(child, HasLoaderAdapter):
            loader = _scoped(path, child.adapt_loader)(loader)
    return loader


# ---------------------------------------------------------------------------
# Convenience: build a WeightLoader over a flat dict
# ---------------------------------------------------------------------------


def dict_loader(state: dict[str, WeightData]) -> WeightLoader:
    """Wraps a flat ``dict[str, WeightData]`` as a :class:`WeightLoader`.

    Useful for tests and for callers that already have a materialised
    flat state dict. For loading from disk, prefer
    :func:`_loader_over_weights` so resolution stays lazy.

    Args:
        state: Flat state dict.

    Returns:
        A :class:`WeightLoader` that resolves names by dict lookup and
        enumerates by prefix filter.
    """
    return _LoaderImpl(
        call=lambda name: state[name],
        keys_fn=lambda p: (k for k in state if k.startswith(p)),
    )


def _loader_over_weights(w: Weights) -> WeightLoader:
    """Wraps a :class:`~max.graph.weights.Weights` source as a :class:`WeightLoader`.

    Defers tensor materialisation: ``w[name].data()`` is called only when
    the wrapped loader is queried, so the safetensors mmap stays cold for
    parameters the Module never asks for. ``keys`` is served from the
    source's index via :meth:`Weights.items`.

    Args:
        w: A loaded :class:`~max.graph.weights.Weights` (typically the
            return value of :func:`~max.graph.weights.load_weights`).

    Returns:
        A :class:`WeightLoader` over ``w``'s namespace.
    """
    return _LoaderImpl(
        call=lambda name: w[name].data(),
        keys_fn=lambda p: (name for name, _ in w.items() if name.startswith(p)),
    )


def _role_prefixed_loader(
    per_role: dict[str, WeightLoader],
) -> WeightLoader:
    """Unions per-role loaders into one, routing ``"role.X"`` queries.

    Queries of the form ``f"{role}.{name}"`` resolve via
    ``per_role[role](name)``; ``keys(prefix)`` re-prefixes each sub-loader's
    enumeration and pushes the un-prefixed remainder down so a query like
    ``keys("transformer.blocks.")`` only walks the matching role.

    Args:
        per_role: Mapping from role string to sub-loader. Each sub-loader's
            namespace is exposed under that role's dotted prefix.

    Returns:
        A :class:`WeightLoader` over the role-prefixed union.
    """

    def call(name: str) -> WeightData:
        role, _, rest = name.partition(".")
        if rest and role in per_role:
            return per_role[role](rest)
        raise KeyError(name)

    def keys_fn(p: str = "") -> Iterable[str]:
        for role, loader in per_role.items():
            dotted = f"{role}."
            if p.startswith(dotted):
                for k in loader.keys(p[len(dotted) :]):
                    yield f"{dotted}{k}"
            elif not p or dotted.startswith(p):
                for k in loader.keys():  # noqa: SIM118 (WeightLoader.keys)
                    yield f"{dotted}{k}"

    return _LoaderImpl(call=call, keys_fn=keys_fn)
