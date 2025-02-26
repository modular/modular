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
from dataclasses import dataclass
from functools import wraps
from inspect import signature
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Mapping,
    Tuple,
    Union,
    get_args,
)

import numpy as np
from max.driver import DLPackArray, Tensor
from max.dtype import DType
from max.graph import DeviceRef, ShapeLike, TensorValue, Weight
from max.graph.quantization import QuantizationEncoding
from max.graph.weights import WeightData

from .._identity import IdentitySet

DLPackCompatible = Union[DLPackArray, np.ndarray]


@dataclass
class ShardingStrategy:
    """Defines how to load and shard a weight onto multiple devices."""

    host_device: DeviceRef
    shard_value: Callable[[Weight], tuple[TensorValue, ...]]


class Layer:
    """Base Layer class.

    Currently, only functionality is for adding hooks to the call function of
    each layer to support testing, debugging or profiling.
    """

    def __init_subclass__(cls):
        if cls.__name__ == "LayerV2":
            # LayerV2 subclasses Layer, but we don't want to apply
            # _call_with_hooks to it.
            return
        # Check `__dict__` instead of `hasattr` because `hasattr` passes on
        # subclasses that don't implement the method.
        if "__call__" in cls.__dict__:
            setattr(cls, "__call__", _call_with_hooks(cls.__dict__["__call__"]))

    def __call__(self, *args, **kwargs):
        """Defines the forward function of this layer.

        Subclasses must override this function. There is no exact signature that a
        call function must follow, but inputs/outputs should generally be
        `max.graph.TensorValue`. Non-`TensorValue` inputs are fine, but
        cannot be updated once the graph is built.
        """


class LayerV2(Layer, ABC):
    """(new) Base class for model layers with weight and device management.

    This will be merged with the above class once all layers have been moved to
    V2.
    """

    def __init__(self):
        # `__init__` may be called if `__setattr__` is called before
        # `super().__init__()`. So, to avoid resetting the values, first
        # check to see if the layer has been initialized before.
        if not hasattr(self, "_sublayers"):
            self._sublayers: dict[str, LayerV2] = {}
            self._layer_weights: dict[str, Weight] = {}
            self._weight_values: dict[str, DLPackCompatible] = {}
            self._shared_weights: dict[str, Weight] = {}

            # Update device list and sharding strategy to a singular object
            # similar to a partition spec.
            self._devices: tuple[DeviceRef, ...] = ()
            self._sharding_strategy: ShardingStrategy | None = None

    def __setattr__(self, name, value):
        try:
            if isinstance(value, LayerV2):
                self._sublayers[name] = value
                if self._devices or self._sharding_strategy:
                    value.to(
                        *self._devices,
                        sharding_strategy=self._sharding_strategy,
                    )
            elif isinstance(value, Weight):
                self._layer_weights[value.name] = value
        except AttributeError:
            # The layer didn't call `super().__init__()` first thing.
            LayerV2.__init__(self)
            self.__setattr__(name, value)
            return
        super().__setattr__(name, value)

    def __repr__(self):
        # TODO: Make this pretty
        return f"{type(self).__name__}({len(self.sublayers)} layers, {len(self.layer_weights)} weights)"

    def to(
        self,
        *devices: DeviceRef,
        sharding_strategy: ShardingStrategy | None = None,
    ) -> None:
        if len(self._devices) > 1 and not sharding_strategy:
            raise ValueError(
                "Must provide a sharding strategy if multiple  devices are provided."
            )

        for _, layer in recursive_named_layers(self):
            layer._devices = devices
            layer._sharding_strategy = sharding_strategy

    @property
    def layer_weights(self) -> dict[str, Weight]:
        return self._layer_weights

    def __delattr__(self, name: str):
        self._sublayers.pop(name, None)
        self._layer_weights.pop(name, None)
        self._shared_weights.pop(name, None)
        super().__delattr__(name)

    def set_shared_weight(self, name: str, weight: Weight):
        setattr(self, name, weight)
        self._shared_weights[name] = weight

    @property
    def sublayers(self) -> dict[str, LayerV2]:
        return self._sublayers

    def load_state_dict(
        self,
        state_dict: Mapping[str, DLPackCompatible | WeightData],
        *,
        override_quantization_encoding: bool = False,
    ) -> None:
        """Sets the values of all weights in this model.

        Args:
            state_dict: A map from weight name to a numpy array or
                max.driver.Tensor.
            override_quantization_encoding: Whether to override the weight
                quantization based on the loaded value.

        Raises:
            Error if any weight in the model is not present in the state dict.
        """
        for name, layer in recursive_named_layers(self):
            weight_prefix = f"{name}." if name else ""
            for weight_name, weight in layer.layer_weights.items():
                # Skip the shared weights, since their values are loaded with
                # the original layers.
                if weight_name in layer._shared_weights:
                    continue
                full_weight_name = f"{weight_prefix}{weight_name}"
                if (data := state_dict.get(full_weight_name)) is not None:
                    if isinstance(data, WeightData):
                        data = _array_from_weight_loader(
                            weight,
                            data,
                            override_quantization_encoding,
                            full_weight_name,
                        ).data
                    else:
                        _validate_weight_value(weight, data, full_weight_name)
                    self._weight_values[full_weight_name] = data
                    weight.name = full_weight_name
                else:
                    msg = f"Could not find weight '{full_weight_name}'. "
                    if possible_match := difflib.get_close_matches(
                        full_weight_name, state_dict.keys(), n=1
                    ):
                        msg += f" Did you mean '{possible_match[0]}'?"
                    raise ValueError()

    def state_dict(
        self, auto_initialize: bool = True
    ) -> dict[str, DLPackCompatible]:
        """Returns values of all weights in the model.

        The values returned are the same as the values set in `load_state_dict`.
        If `load_state_dict` has not been called and none of the weights have
        values, then they are initialized to zero.

        Args:
            auto_initialize: Determines whether to initialize weights to zero if
                the weight value has not been loaded. If this is False, a
                ValueError is raised if an uninitialized weight is found.

        Returns:
            Map from weight name to the weight value (can be numpy array or
            max.driver.Tensor).
        """

        state_dict = {}
        for full_weight_name, weight in self.raw_state_dict().items():
            if (data := self._weight_values.get(full_weight_name)) is None:
                if not auto_initialize:
                    raise ValueError(
                        f"Weight '{full_weight_name}' was not initialized."
                    )
                data = self._weight_values[full_weight_name] = Tensor.zeros(
                    weight.shape.static_dims, weight.dtype
                )
            state_dict[full_weight_name] = data
        return state_dict

    def raw_state_dict(self) -> dict[str, Weight]:
        """Returns all weights objects in the model.
        Unlike `state_dict`, this returns `max.graph.Weight` objects instead of
        the assigned values. Some parameters inside the `Weight` can be
        configured before a graph is built. Do not change these attributes after
        building a graph:

          - Weight.shape
          - Weight.dtype
          - Weight.quantization_encoding
          - Weight.device
          - Weight.align

        Returns:
            Map from weight name to the `max.graph.Weight` object.
        """
        state_dict = {}
        for name, layer in recursive_named_layers(self):
            prefix = f"{name}." if name else ""

            for weight_name, weight in layer.layer_weights.items():
                state_dict[f"{prefix}{weight_name}"] = weight
        return state_dict

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """Defines the forward function of this layer.

        Subclasses must override this function. There is no exact signature that a
        call function must follow, but inputs/outputs should generally be
        `max.graph.TensorValue`. Non-`TensorValue` inputs are fine, but
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
        data = data.view(DType.uint8)
    else:
        data = data.astype(weight.dtype)

    if weight.quantization_encoding:
        # TODO: Set the quantized weight shape correctly when initializing the
        # weight. For now, we trust that the value loaded from the checkpoint
        # has the correct shape.
        weight._shape = data.shape
    elif weight.shape != data.shape:
        raise ValueError(
            f"Value provided to weight '{name}' had different shape"
            f" (expected={weight.shape}, actual={data.shape})"
        )

    if weight.quantization_encoding != data.quantization_encoding:
        if override_quantization_encoding:
            weight.quantization_encoding = data.quantization_encoding
        # We don't raise an error if `override_quantization_encoding` is `False`
        # because in some cases the data is not aware of its own quantization
        # type (e.g. data loaded from GPTQ Safetensors do not have a
        # quantization label)
    return data.astype(weight.dtype)


def _get_value_shape_dtype(value: DLPackCompatible) -> tuple[ShapeLike, DType]:
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


def _validate_weight_value(weight: Weight, value: Any, name: str) -> None:
    if not isinstance(value, get_args(DLPackCompatible)):
        raise ValueError(
            f"The class type of '{name}' value ({type(value)}) is not an array "
            "type that we understand. Please use a numpy array or max.driver.Tensor."
        )

    shape, dtype = _get_value_shape_dtype(value)

    diffs = []
    weight_shape = tuple(weight.shape.static_dims)
    if shape != weight_shape:
        diffs.append(f"shape (expected={weight_shape}, actual={shape})")
    if dtype != weight.dtype:
        diffs.append(f"dtype (expected={weight.dtype}, actual={dtype})")
    if diffs:
        diff_str = " and ".join(diffs)
        raise ValueError(
            f"Value provided to weight '{name}' had different {diff_str}."
        )


def recursive_named_layers(
    parent: LayerV2, prefix: str = ""
) -> Iterable[tuple[str, LayerV2]]:
    """Recursively walks through the layers and generates names."""
    seen = IdentitySet()
    queue: deque[tuple[str, LayerV2]] = deque()
    queue.append((prefix, parent))

    while queue:
        name, layer = queue.popleft()
        if layer in seen:
            continue
        seen.add(layer)

        yield (name, layer)
        prefix = f"{name}." if name else ""
        for local_name, layer in layer.sublayers.items():
            queue.append((f"{prefix}{local_name}", layer))


_LOCAL = threading.local()
_LAYER_HOOKS = _LOCAL._layer_hooks = []


def add_layer_hook(
    fn: Callable[[Layer, Tuple[Any, ...], Dict[str, Any], Any], Any],
) -> None:
    """Adds a hook to call a function after each layer's `__call__`.

    The function will be passed four inputs: the layer, input_args,
    input_kwargs and outputs. The function can either return `None` or new
    outputs that will replace the layer returned outputs.

    Note that input and outputs contain graph Values, which show limited
    information (like shape and dtype). You can still see the computed values
    if you include the Value in the `graph.output` op, or call `value.print`.

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


def clear_hooks():
    """Remove all hooks."""
    _LAYER_HOOKS.clear()


def _call_with_hooks(call_fn):
    @wraps(call_fn)
    def __call_with_hooks(layer, *args, **kwargs):
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
