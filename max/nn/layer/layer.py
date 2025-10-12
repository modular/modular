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

from __future__ import annotations

import difflib
import threading
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import wraps
from inspect import signature
from itertools import islice
from typing import Any, Protocol

import numpy as np
from max.driver import DLPackArray, Tensor
from max.dtype import DType
from max.graph import (
    DeviceRef,
    Graph,
    Shape,
    ShapeLike,
    ShardingStrategy,
    StaticDim,
    Type,
    Value,
    Weight,
)
from max.graph.quantization import QuantizationEncoding
from max.graph.weights import WeightData
from typing_extensions import Self

from .._identity import IdentitySet


class Shardable(Protocol):
    """Protocol for objects that support sharding across multiple devices.

    This protocol defines the interface that all shardable components
    (like Linear layers and Weight objects) must implement to participate
    in distributed computation.
    """

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Gets the weight sharding strategy."""
        ...

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Sets the weight sharding strategy.

        Args:
            strategy: A ShardingStrategy that defines how to shard the weight.
        """
        ...

    def shard(self, devices: Iterable[DeviceRef]) -> Sequence[Self]:
        """Creates a sharded view of this object for a specific device.

        Args:
            device: The devices where this shard should reside.

        Returns:
            A sequence of sharded instances of this object.
        """
        ...


class Layer:
    """
    .. deprecated:: 25.2.

    Base class for neural network components.
    Use :obj:`Module` instead.

    Provides functionality for adding hooks to the call function of
    each layer to support testing, debugging or profiling.
    """

    def __init_subclass__(cls):
        if cls.__name__ == "Module":
            # Module subclasses Layer, but we don't want to apply
            # _call_with_hooks to it.
            return
        # Check `__dict__` instead of `hasattr` because `hasattr` passes on
        # subclasses that don't implement the method.
        if "__call__" in cls.__dict__:
            setattr(cls, "__call__", _call_with_hooks(cls.__dict__["__call__"]))  # noqa: B010

    def __call__(self, *args, **kwargs):
        """Defines the forward function of this layer.

        Subclasses must override this function. There is no exact signature that a
        call function must follow, but inputs/outputs should generally be
        `max.graph.TensorValue`. Non-`TensorValue` inputs are fine, but
        cannot be updated once the graph is built.
        """


class Module(Layer, ABC):
    """Base class for model components with weight management.

    Provides functionality to create custom layers and construct networks with automatic weight tracking.

    The following example uses the :obj:`Module` class to create custom layers and build a neural network:

    .. code-block:: python

        from max import nn
        from max.dtype import DType
        from max.graph import Weight, ops, DeviceRef

        class Linear(nn.Module):
            def __init__(self, in_dims, out_dims):
                super().__init__()
                self.weight = Weight("weight", DType.float32, (in_dim, out_dim), DeviceRef.CPU())

            def __call__(self, x):
                return x @ self.weight.T

        class MLP(nn.Module):
            def __init__(self):
                self.up = Linear(5, 10)
                self.gate = Linear(5, 10)
                self.down = Linear(10, 5)

            def __call__(self, x):
                return self.down(ops.silu(self.gate(x)) + self.up(x))

        model = MLP()
        print(model.state_dict())  # {"up.weight": Tensor([5, 10]), ...}

    Constructing a graph without :obj:`Module` can result in name collisions
    with the weights (in this example, there would be three weights with the
    name `Weight`). With :obj:`Module`, you can use :obj:`state_dict()` or
    :obj:`load_state_dict()` to initialize or set the weights values, and finalize
    the weight names to be unique within the model.
    """

    def __init__(self) -> None:
        # `__init__` may be called if `__setattr__` is called before
        # `super().__init__()`. So, to avoid resetting the values, first
        # check to see if the layer has been initialized before.
        if not hasattr(self, "_sublayers"):
            self._sublayers: dict[str, Module] = {}
            self._layer_weights: dict[str, Weight] = {}
            self._weight_values: dict[str, DLPackArray] = {}
            self._shared_weights: dict[str, Weight] = {}

    def __setattr__(self, name: str, value: Any) -> None:
        try:
            if isinstance(value, Module):
                self._sublayers[name] = value
            elif isinstance(value, Weight):
                if existing_weight := getattr(self, name, None):
                    if isinstance(existing_weight, Weight):
                        # If the attribute being set is a weight, remove the existing weight from
                        # the layer weights. This is to avoid having irrelevant weights returned by
                        # `raw_state_dict()`. This is particularly important for subgraphs as Weight
                        # instances are materialized by accessing their `_mlir_value` attribute, and
                        # doing so on an weight that's meant to be sharded will raise an error because
                        # we'll end up trying to add a weight that's already been added to the graph.
                        del self._layer_weights[existing_weight.name]
                self._layer_weights[value.name] = value
        except AttributeError:
            # The layer didn't call `super().__init__()` first thing.
            Module.__init__(self)
            self.__setattr__(name, value)
            return
        super().__setattr__(name, value)

    def __repr__(self) -> str:
        # TODO: Make this pretty
        return f"{type(self).__name__}({len(self.sublayers)} layers, {len(self.layer_weights)} weights)"

    @property
    def layer_weights(self) -> dict[str, Weight]:
        return self._layer_weights

    def __delattr__(self, name: str) -> None:
        self._sublayers.pop(name, None)
        self._layer_weights.pop(name, None)
        self._shared_weights.pop(name, None)
        super().__delattr__(name)

    def set_shared_weight(self, name: str, weight: Weight) -> None:
        setattr(self, name, weight)
        self._shared_weights[name] = weight

    def build_subgraph(
        self,
        name: str,
        input_types: Sequence[Type[Any] | list[Type[Any]]],
        weight_prefix: str = "",
    ) -> Graph:
        """Builds a subgraph for this module.

        This method creates a subgraph that encapsulates the module's logic,
        handling input types, weights, and creating a graph with the module's
        computation.

        Once the subgraph is built, it can be called using the :obj:`ops.call`
        op.

        Args:
            name: The name of the subgraph to create.
            input_types: A list of input types for the subgraph. Each element can be
                either a single :obj:`Type` or a list of :obj:`Type` objects.
            weight_prefix: Optional prefix for weight names in the subgraph. If provided,
                weights with names starting with this prefix will have their names
                modified by removing the prefix and will be marked as placeholders.

        Returns:
            :obj:`Graph`: The created subgraph containing the module's computation.

        Note:
            - Placeholder weights will require the :obj:`prefix` attribute of :obj:`ops.call` to be set.
        """
        layer_weights = list(self.raw_state_dict().values())
        subgraph_input_types: list[Type[Any]] = []

        def flatten(t: Any, result: list[Any]) -> None:
            if isinstance(t, list | tuple):
                for item in t:
                    flatten(item, result)
            else:
                result.append(t)

        def take(it: Iterable[Value[Any]], n: int) -> list[Value[Any]]:
            """Return the next *n* items from *it* as a list."""
            return list(islice(it, n))

        flatten(input_types, subgraph_input_types)
        with Graph.current.add_subgraph(
            name, input_types=subgraph_input_types
        ) as subgraph:
            subgraph_inputs = []
            inputs = iter(subgraph.inputs)

            for input_type in input_types:
                if isinstance(input_type, list):
                    subgraph_inputs.append(take(inputs, len(input_type)))
                else:
                    subgraph_inputs.append(next(inputs))

            if weight_prefix:
                for weight in filter(
                    lambda w: w.name.startswith(weight_prefix), layer_weights
                ):
                    weight._placeholder = True
                    weight.name = weight.name.removeprefix(weight_prefix)

            result = self(*subgraph_inputs)
            if isinstance(result, list | tuple):
                subgraph.output(*result)
            else:
                subgraph.output(result)

        return subgraph

    @property
    def sublayers(self) -> dict[str, Module]:
        return self._sublayers

    def load_state_dict(
        self,
        state_dict: Mapping[str, DLPackArray | WeightData],
        *,
        override_quantization_encoding: bool = False,
        weight_alignment: int | None = None,
        strict: bool = True,
    ) -> None:
        """Sets the values of all weights in this model.

        Args:
            state_dict: A map from weight name to a numpy array or
                :obj:`max.driver.Tensor`.
            override_quantization_encoding: Whether to override the weight
                quantization based on the loaded value.
            weight_alignment: If specified, overrides the alignment for each
                weight in the `Module`. If left as `None`, each value in
                state_dict must be aligned by the default dtype alignment.
            strict: If True, raises an error if any weights required by the
                `Module` are missing from `state_dict`, or if any keys in
                `state_dict` were not used by the `Module`. If False, both
                missing and unexpected keys are tolerated and reported only
                via return values/logging by callers.

        Raises:
            ValueError: If `strict` is True and any required weight is missing
                from `state_dict`, or if `state_dict` contains keys not used by
                the `Module`.
        """
        loaded_keys = set()
        missing_keys = set()

        for name, layer in recursive_named_layers(self):
            weight_prefix = f"{name}." if name else ""
            for weight_name, weight in layer.layer_weights.items():
                # Skip shared weights, as they are loaded with the original layers.
                if weight_name in layer._shared_weights:
                    continue

                full_weight_name = f"{weight_prefix}{weight_name}"

                if (data := state_dict.get(full_weight_name)) is not None:
                    loaded_keys.add(full_weight_name)
                    if isinstance(data, WeightData):
                        data = _array_from_weight_loader(
                            weight,
                            data,
                            override_quantization_encoding,
                            full_weight_name,
                        ).data
                    else:
                        _validate_weight_value(weight, data, full_weight_name)

                    if weight_alignment:
                        weight.align = weight_alignment

                    _check_alignment(
                        data,
                        weight.align or weight.dtype.align,
                        full_weight_name,
                    )
                    self._weight_values[full_weight_name] = data
                    weight.name = full_weight_name
                else:
                    # If a key is missing, just add it to the set for later.
                    missing_keys.add(full_weight_name)

        # After the loop, check for all errors at once if in strict mode.
        unused_keys = state_dict.keys() - loaded_keys

        if strict and (missing_keys or unused_keys):
            parts = []
            if missing_keys:
                sorted_missing = sorted(list(missing_keys))
                parts.append(
                    f"Missing required weights: {', '.join(sorted_missing)}"
                )

                # Add a helpful "Did you mean?" suggestion for the first missing key.
                first_missing_key = sorted_missing[0]
                if possible_match := difflib.get_close_matches(
                    first_missing_key, state_dict.keys(), n=1
                ):
                    parts.append(
                        f"For '{first_missing_key}', did you mean '{possible_match[0]}'?"
                    )

            if unused_keys:
                parts.append(
                    f"Unexpected keys in state_dict: {', '.join(sorted(list(unused_keys)))}"
                )

            raise ValueError(
                "load_state_dict() strict=True validation failed. "
                + "; ".join(parts)
            )

    def state_dict(
        self, auto_initialize: bool = True
    ) -> dict[str, DLPackArray]:
        """Returns values of all weights in the model.

        The values returned are the same as the values set in :obj:`load_state_dict`.
        If :obj:`load_state_dict` has not been called and none of the weights have
        values, then they are initialized to zero.

        Args:
            auto_initialize: Determines whether to initialize weights to zero if
                the weight value has not been loaded. If this is False, a
                ValueError is raised if an uninitialized weight is found.

        Returns:
            Map from weight name to the weight value (can be numpy array or
            :obj:`max.driver.Tensor`).
        """

        state_dict = {}
        for full_weight_name, weight in self.raw_state_dict().items():
            if (data := self._weight_values.get(full_weight_name)) is None:
                if not auto_initialize:
                    raise ValueError(
                        f"Weight '{full_weight_name}' was not initialized."
                    )
                # Contents of weights should be filled with zeros.
                data = self._weight_values[full_weight_name] = Tensor.zeros(
                    shape=weight.shape.static_dims, dtype=weight.dtype
                )
            state_dict[full_weight_name] = data
            weight.name = full_weight_name
        return state_dict

    def raw_state_dict(self) -> dict[str, Weight]:
        """Returns all weights objects in the model.
        Unlike :obj:`state_dict`, this returns :obj:`max.graph.Weight` objects instead of
        the assigned values. Some parameters inside the :obj:`Weight` can be
        configured before a graph is built. Do not change these attributes after
        building a graph:

        - :obj:`~max.graph.Weight.align`
        - :obj:`~max.graph.Weight.dtype`
        - :obj:`~max.graph.Weight.quantization_encoding`
        - :obj:`~max.graph.Weight.shape`

        Returns:
            Map from weight name to the :obj:`max.graph.Weight` object.
        """
        state_dict = {}
        for name, layer in recursive_named_layers(self):
            prefix = f"{name}." if name else ""

            for weight_name, weight in layer.layer_weights.items():
                if weight_name in layer._shared_weights:
                    continue
                state_dict[f"{prefix}{weight_name}"] = weight
        return state_dict

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """Defines the forward function of this layer.

        Subclasses must override this function. There is no exact signature that a
        call function must follow, but inputs/outputs should generally be
        :obj:`max.graph.TensorValue`. Non-:obj:`TensorValue` inputs are fine, but
        cannot be updated once the graph is built.
        """


def _array_from_weight_loader(
    weight: Weight,
    data: WeightData,
    override_quantization_encoding: bool,
    name: str,
) -> WeightData:
    """Processes and validates the data from WeightData."""

    if weight.quantization_encoding == QuantizationEncoding.GPTQ:
        # Load all weights with GPTQ quantization as uint8.
        # Store the original shape and dtype of the weight (used in layers like
        # GPTLinear).
        weight.original_dtype_and_shape = (data.dtype, data.shape)
        data.data = new_data = Tensor.from_dlpack(data.data).view(DType.uint8)
        data.dtype = DType.uint8
        data.shape = Shape(new_data.shape)
        weight._shape = Shape(new_data.shape)

    if weight.quantization_encoding:
        # TODO: Set the quantized weight shape correctly when initializing the
        # weight. For now, we trust that the value loaded from the checkpoint
        # has the correct shape.
        weight._shape = data.shape
    elif (weight.shape == [] and data.shape == [1]) or (
        weight.shape == [1] and data.shape == []
    ):
        # These shapes are actually the same.
        # Treat the data as if it has the correct shape.
        data.shape = Shape(weight._shape)
    elif weight.shape != data.shape:
        raise ValueError(
            f"Value provided to weight '{name}' had different shape"
            f" (expected={weight.shape}, actual={data.shape})"
        )

    if weight.quantization_encoding != data.quantization_encoding:
        if (
            override_quantization_encoding
            and data.quantization_encoding is not None
        ):
            weight.quantization_encoding = data.quantization_encoding
        # We don't raise an error if `override_quantization_encoding` is `False`
        # because in some cases the data is not aware of its own quantization
        # type (e.g. data loaded from GPTQ Safetensors do not have a
        # quantization label)

    if weight.dtype != data.dtype:
        raise ValueError(
            f"Value provided to weight '{name}' had different dtype"
            f" (expected={weight.dtype}, actual={data.dtype})"
        )

    return data


def _get_value_shape_dtype(value: DLPackArray) -> tuple[ShapeLike, DType]:
    if isinstance(value, Tensor):
        shape = value.shape
        dtype = value.dtype
    elif isinstance(value, np.ndarray):
        shape = value.shape
        dtype = DType.from_numpy(value.dtype)
    else:
        # `from_dlpack` does not copy the data.
        value_tensor = Tensor.from_dlpack(value)
        shape = value_tensor.shape
        dtype = value_tensor.dtype

    return shape, dtype


def _check_alignment(value: DLPackArray, align: int, name: str) -> None:
    # Fast path for ndarray.
    # The use of Tensor.from_dlpack always copies if the numpy array is not
    # writeable, which is very common for weight values.
    #
    # This logic special cases the two code paths that potentially could copy
    # and performs the alignment check manually.
    if isinstance(value, np.ndarray):
        data = value.ctypes.data
        if data % align == 0:
            return
    elif isinstance(value, Tensor):
        if value._aligned(align):
            return
    else:
        tensor = Tensor.from_dlpack(value)
        if tensor._aligned(align):
            return

    raise ValueError(
        f"Found unaligned weight '{name}' (expected alignment={align})."
        "If you are using a Safetensor checkpoint, it is recommended that "
        "you copy the weight to correct the alignment, or pass "
        "`weight_alignment=1` to `Module.load_state_dict()`."
    )


def _validate_weight_value(
    weight: Weight, value: DLPackArray, name: str
) -> None:
    if not isinstance(value, DLPackArray):
        raise ValueError(
            f"The class type of '{name}' value ({type(value)}) is not an array "
            "type that we understand. Please use a numpy array or max.driver.Tensor."
        )

    shape, dtype = _get_value_shape_dtype(value)

    diffs = []

    # Check if weight has symbolic dimensions
    # Convert weight.shape to list to ensure it's sized
    weight_shape_dims = list(weight.shape)
    weight_has_symbolic_dims = len(weight_shape_dims) != len(
        weight.shape.static_dims
    )

    # Convert shape to tuple to ensure it's sized
    shape_tuple = tuple(shape)

    if weight_has_symbolic_dims:
        # For weights with symbolic dimensions, validate by comparing static dimensions
        # at their correct positions, allowing symbolic dimensions to vary
        if len(shape_tuple) != len(weight_shape_dims):
            # Shape rank must match
            diffs.append(
                f"shape rank (expected={len(weight_shape_dims)}, actual={len(shape_tuple)})"
            )
        else:
            # Check each dimension: static dims must match, symbolic dims can vary
            mismatches = []
            for i, (weight_dim, value_dim) in enumerate(
                zip(weight_shape_dims, shape_tuple, strict=True)
            ):
                if isinstance(weight_dim, StaticDim):
                    # This is a static dimension - must match exactly
                    if int(weight_dim) != value_dim:
                        mismatches.append(
                            f"dim[{i}]: expected {int(weight_dim)}, got {value_dim}"
                        )
                # Symbolic dimensions are allowed to vary, so no check needed

            if mismatches:
                diffs.append(f"shape ({', '.join(mismatches)})")
    else:
        # For fully static weights, use the original validation
        weight_shape = tuple(weight.shape.static_dims)
        if shape_tuple != weight_shape:
            diffs.append(
                f"shape (expected={weight_shape}, actual={shape_tuple})"
            )

    if dtype != weight.dtype:
        diffs.append(f"dtype (expected={weight.dtype}, actual={dtype})")
    if diffs:
        diff_str = " and ".join(diffs)
        raise ValueError(
            f"Value provided to weight '{name}' had different {diff_str}."
        )


def recursive_named_layers(
    parent: Module, prefix: str = ""
) -> Iterable[tuple[str, Module]]:
    """Recursively walks through the layers and generates names."""
    seen = IdentitySet[Module]()
    queue: deque[tuple[str, Module]] = deque()
    queue.append((prefix, parent))

    while queue:
        name, layer = queue.popleft()
        if layer in seen:
            continue
        seen.add(layer)

        yield (name, layer)
        prefix = f"{name}." if name else ""
        for local_name, layer in layer.sublayers.items():  # noqa: B020
            queue.append((f"{prefix}{local_name}", layer))


_LOCAL = threading.local()
_LAYER_HOOKS = _LOCAL._layer_hooks = []


def add_layer_hook(
    fn: Callable[[Layer, tuple[Any, ...], dict[str, Any], Any], Any],
) -> None:
    """Adds a hook to call a function after each layer's ``__call__``.

    The function will be passed four inputs:
    - layer
    - input_args
    - input_kwargs
    - outputs

    The function can either return `None` or new
    outputs that will replace the layer returned outputs.

    Note that input and outputs contain graph Values, which show limited
    information (like :obj:`~max.graph.TensorValue.shape` and :obj:`~max.graph.TensorValue.dtype`). You can still see the computed values
    if you include the Value in the :obj:`graph.ops.output` op, or call :obj:`graph.ops.print`.

    Example of printing debug inputs:

    .. code-block:: python

        def print_info(layer, args, kwargs, outputs):
            print("Layer:", type(layer).__name__)
            print("Input args:", args)
            print("Input kwargs:", kwargs)
            print("Outputs:", outputs)
            return outputs

        add_layer_hook(print_info)
    """
    _LAYER_HOOKS.append(fn)


def clear_hooks() -> None:
    """Remove all hooks."""
    _LAYER_HOOKS.clear()


def _call_with_hooks(call_fn: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(call_fn)
    def __call_with_hooks(layer: Layer, *args, **kwargs) -> Any:
        # Hide this wrapper from rich traceback.
        _rich_traceback_omit = True

        outputs = call_fn(layer, *args, **kwargs)
        # Use the inspect lib to ensure that args and kwargs are passed
        # to the hook as defined in the function signature.
        bound_args = signature(call_fn).bind(layer, *args, **kwargs)
        for hook in _LAYER_HOOKS:
            # Call the hook. Note that the first argument in `bound_args.args`
            # is the layer, so it is skipped.
            hook_outputs = hook(
                layer, bound_args.args[1:], bound_args.kwargs, outputs
            )
            if hook_outputs is not None:
                outputs = hook_outputs
        return outputs

    return __call_with_hooks
