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
"""Test the state_dict interface."""

import numpy as np
import pytest
from max.dtype import DType
from max.graph import DeviceRef, Weight
from max.nn.layer.layer import Module
from max.nn.layer.layer_list import LayerList


# Dummy module for testing load_state_dict
class SimpleTestModule(Module):
    def __init__(self) -> None:
        super().__init__()

        self.fc1_weight = Weight(
            "fc1.weight",
            dtype=DType.float32,
            shape=[2, 2],
            device=DeviceRef.CPU(),
        )
        self.fc1_bias = Weight(
            "fc1.bias", dtype=DType.int32, shape=[2], device=DeviceRef.CPU()
        )

    def __call__(self, x):  # noqa: ANN001
        return x


def test_load_state_dict_strict_true() -> None:
    """Tests load_state_dict with strict=True, expecting an error for extra keys."""
    module = SimpleTestModule()

    fc1_weight_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    fc1_bias_data = np.array([5, 6], dtype=np.int32)
    extra_key_data = np.array([7.0, 8.0], dtype=np.float32)

    state_dict_with_extra = {
        "fc1.weight": fc1_weight_data,
        "fc1.bias": fc1_bias_data,
        "extra.key": extra_key_data,  # This key is not in SimpleTestModule
    }

    with pytest.raises(ValueError, match="Unexpected keys in state_dict"):
        module.load_state_dict(state_dict_with_extra, strict=True)


def test_load_state_dict_strict_false() -> None:
    """Tests load_state_dict with strict=False, allowing and ignoring extra keys."""
    module = SimpleTestModule()

    fc1_weight_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    fc1_bias_data = np.array([5, 6], dtype=np.int32)
    extra_key_data = np.array([7.0, 8.0], dtype=np.float32)

    state_dict_with_extra = {
        "fc1.weight": fc1_weight_data,
        "fc1.bias": fc1_bias_data,
        "extra.key": extra_key_data,
    }

    module.load_state_dict(state_dict_with_extra, strict=False)

    # Verify that the known weights are loaded correctly
    loaded_dict = module.state_dict()

    assert "fc1.weight" in loaded_dict
    np.testing.assert_array_equal(
        np.asarray(loaded_dict["fc1.weight"]), fc1_weight_data
    )

    assert "fc1.bias" in loaded_dict
    np.testing.assert_array_equal(
        np.asarray(loaded_dict["fc1.bias"]), fc1_bias_data
    )

    # Verify that the extra key was ignored and is not in the module's state_dict
    assert "extra.key" not in loaded_dict
    assert len(loaded_dict) == 2


def test_load_state_dict_missing_key() -> None:
    """Tests load_state_dict with, expecting an error for missing keys."""
    module = SimpleTestModule()

    fc1_weight_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    # State dict missing 'fc1.bias'
    state_dict_missing_key = {
        "fc1.weight": fc1_weight_data,
    }

    with pytest.raises(ValueError, match="Missing required weights"):
        module.load_state_dict(state_dict_missing_key)


class _Leaf(Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = Weight(
            "weight", dtype=DType.float32, shape=[2], device=DeviceRef.CPU()
        )

    def __call__(self, x):  # noqa: ANN001
        return x


class _NameOmittingContainer(Module):
    def __init__(self) -> None:
        super().__init__()
        self.inner = _Leaf()

    @property
    def _omit_module_attr_name(self) -> bool:
        return True

    def __call__(self, x):  # noqa: ANN001
        return x


class _Tree(Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = LayerList([_Leaf(), _Leaf()])
        self.flat = _NameOmittingContainer()
        # Reachable a second time through `alias`; must be visited once.
        self.alias = self.layers[0]

    def __call__(self, x):  # noqa: ANN001
        return x


def test_named_modules_prefixes() -> None:
    """Tests that named_modules yields weight prefixes, not display paths."""
    tree = _Tree()
    prefix_of = {id(module): prefix for prefix, module in tree.named_modules()}

    assert prefix_of[id(tree)] == ""
    assert prefix_of[id(tree.layers)] == "layers."
    assert prefix_of[id(tree.layers[1])] == "layers.1."
    # A module reachable twice takes the shallower path, as its weight FQNs do.
    assert prefix_of[id(tree.layers[0])] == "alias."
    # The name-omitting container shares its parent's prefix, so its child
    # is `inner.`, not `flat.inner.`.
    assert prefix_of[id(tree.flat)] == ""
    assert prefix_of[id(tree.flat.inner)] == "inner."


def test_named_modules_visits_each_module_once() -> None:
    """Tests that a module reachable through two attributes is yielded once."""
    tree = _Tree()
    modules = [module for _, module in tree.named_modules()]
    assert sum(module is tree.layers[0] for module in modules) == 1


def test_named_modules_agrees_with_raw_state_dict() -> None:
    """Tests that prefix + local weight name reproduces every FQN."""
    tree = _Tree()
    from_walk = {
        f"{prefix}{name}"
        for prefix, module in tree.named_modules()
        for name in module.layer_weights
    }
    assert from_walk == set(tree.raw_state_dict())
