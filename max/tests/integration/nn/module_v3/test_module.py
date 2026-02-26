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
"""Tests for max.nn.Module."""

from __future__ import annotations

import re
import weakref

import pytest
from max import driver
from max.driver import CPU, Accelerator, accelerator_count
from max.dtype import DType
from max.experimental import functional as F
from max.experimental import random
from max.experimental.tensor import Tensor, TensorType, defaults
from max.nn.module_v3.module import Module, module_dataclass


@module_dataclass
class SubModule(Module[[Tensor], Tensor]):
    b: Tensor
    eps: float = 1e-5

    def forward(self, x: Tensor) -> Tensor:
        return x + self.b


@module_dataclass
class TestModule(Module[[Tensor], Tensor]):
    a: Tensor
    sub: SubModule

    def forward(self, x: Tensor) -> Tensor:
        return self.sub(x) + self.a


@module_dataclass
class SuperModule(Module[[Tensor], Tensor]):
    mod: TestModule


@pytest.fixture
def test_module():  # noqa: ANN201
    return TestModule(
        a=Tensor.constant(1),
        sub=SubModule(b=Tensor.constant(2)),
    )


@pytest.fixture
def lazy_test_module():  # noqa: ANN201
    with F.lazy():
        return TestModule(
            a=Tensor.constant(1),
            sub=SubModule(b=Tensor.constant(2)),
        )


@pytest.fixture
def super_module(test_module: TestModule):  # noqa: ANN201
    return SuperModule(mod=test_module)


def test_module_dataclass() -> None:
    @module_dataclass
    class Test(Module[..., None]):
        a: int
        b: int = 0

    assert repr(Test(2)) == "Test(a=2)"
    assert repr(Test(1, 3)) == "Test(a=1, b=3)"


def test_module_repr(test_module: TestModule) -> None:
    assert "TestModule" in repr(test_module)
    assert "SubModule" in repr(test_module)
    assert "a=Tensor" in repr(test_module)
    assert "b=Tensor" in repr(test_module)
    # eps is the default value, shouldn't be present
    assert "eps=" not in repr(test_module)

    sub = SubModule(b=Tensor.constant(2), eps=1e-6)

    assert "SubModule" in repr(sub)
    assert "b=Tensor" in repr(sub)
    assert "eps=" in repr(sub)


def test_module_custom_repr() -> None:
    class Linear(Module[..., None]):
        weight: Tensor
        bias: Tensor | int

        def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
            self.weight = Tensor.zeros([out_dim, in_dim])
            self.bias = Tensor.zeros([out_dim]) if bias else 0

        def __rich_repr__(self):
            out_dim, in_dim = self.weight.shape
            bias = isinstance(self.bias, Tensor)
            yield "in_dim", in_dim
            yield "out_dim", out_dim
            yield "bias", bias, True

    l1 = Linear(2, 2)
    assert repr(l1) == "Linear(in_dim=Dim(2), out_dim=Dim(2))"

    l2 = Linear(3, 1, bias=False)
    assert repr(l2) == "Linear(in_dim=Dim(3), out_dim=Dim(1), bias=False)"


def test_module_decomposition(test_module: TestModule) -> None:
    test_module_2 = TestModule(a=Tensor.constant(1), sub=test_module.sub)
    assert test_module_2.sub is test_module.sub
    assert dict(test_module_2.children) == dict(test_module.children)


def test_module_decomposition_call(test_module: TestModule) -> None:
    x = Tensor.constant(1)
    assert test_module.sub.b.item() == 2
    assert test_module.sub(x).item() == 3


def test_module_forward(test_module: TestModule) -> None:
    x = Tensor.constant(1)
    # __call__ invokes forward, so both should produce the same result
    assert test_module.forward(x).item() == test_module(x).item()


def test_module_local_parameters(test_module: TestModule) -> None:
    assert dict(test_module.local_parameters) == {"a": test_module.a}
    assert dict(test_module.sub.local_parameters) == {"b": test_module.sub.b}


def test_module_parameters(test_module: TestModule) -> None:
    assert dict(test_module.parameters) == {
        "a": test_module.a,
        "sub.b": test_module.sub.b,
    }

    assert dict(test_module.sub.parameters) == {"b": test_module.sub.b}


def test_module_children(
    test_module: TestModule, super_module: SuperModule
) -> None:
    assert dict(super_module.children) == {"mod": test_module}
    assert dict(test_module.children) == {"sub": test_module.sub}
    assert dict(test_module.sub.children) == {}


def test_module_descendants(
    test_module: TestModule, super_module: SuperModule
) -> None:
    assert super_module.mod is test_module
    assert dict(super_module.descendants) == {
        "mod": test_module,
        "mod.sub": test_module.sub,
    }
    assert dict(super_module.mod.descendants) == {"sub": super_module.mod.sub}
    assert dict(test_module.sub.descendants) == {}


def test_apply_to_local_parameters(test_module: TestModule) -> None:
    a = test_module.a
    b = test_module.sub.b

    test_module.apply_to_local_parameters(lambda _, t: t + 1)
    # Applied to a
    assert test_module.a.item() == (a + 1).item()
    # Not applied to submodule
    assert test_module.sub.b.item() == b.item()


def test_apply_to_parameters(test_module: TestModule) -> None:
    a = test_module.a
    b = test_module.sub.b

    test_module.apply_to_parameters(lambda _, t: t + 1)
    # Applied to a
    assert test_module.a.item() == (a + 1).item()
    # Also applied to submodule
    assert test_module.sub.b.item() == (b + 1).item()


def test_apply_to_parameters__qualified_names(test_module: TestModule) -> None:
    names = set()
    expected = dict(test_module.parameters).keys()

    def lookup(name: str, tensor: Tensor):  # noqa: ANN202
        names.add(name)
        return tensor

    test_module.apply_to_parameters(lookup)
    assert expected == names


def test_map_parameters(test_module: TestModule) -> None:
    a = test_module.a
    b = test_module.sub.b

    m2 = test_module.map_parameters(lambda _, t: t + 1)
    # Test parameters were mapped
    assert m2.a.item() == (a + 1).item()
    assert m2.sub.b.item() == (b + 1).item()
    # Not updated in the original module
    assert test_module.a.item() == a.item()
    assert test_module.sub.b.item() == b.item()


def test_load_state_simple_dict(test_module: TestModule) -> None:
    weights = {
        "a": Tensor.constant(5),
        "sub.b": Tensor.constant(6),
    }
    test_module.load_state(lambda name, _: weights[name])
    assert test_module.a.item() == 5
    assert test_module.sub.b.item() == 6


def test_load_state_simple_dict_lookup_failure(test_module: TestModule) -> None:
    weights: dict[str, Tensor] = {}
    # No guarantee on the resulting state here!
    with pytest.raises(KeyError):
        test_module.load_state(lambda name, _: weights[name])


def test_load_state_name_remapping(test_module: TestModule) -> None:
    def remap_name(name: str):  # noqa: ANN202
        name = re.sub(r"\bsub\.", "feed_forward.", name)
        return name

    weights = {
        "a": Tensor.constant(5),
        "feed_forward.b": Tensor.constant(6),
    }

    test_module.load_state(lambda name, _: weights[remap_name(name)])
    assert test_module.a.item() == 5
    assert test_module.sub.b.item() == 6


def test_load_state_dict(test_module: TestModule) -> None:
    weights = {
        "a": Tensor.constant(5),
        "sub.b": Tensor.constant(6),
    }
    test_module.load_state_dict(weights)
    assert test_module.a.item() == 5
    assert test_module.sub.b.item() == 6


def test_load_state_dict_strict(test_module: TestModule) -> None:
    weights = {
        "a": Tensor.constant(5),
        "sub.b": Tensor.constant(6),
        "extra": Tensor.constant(7),
    }
    with pytest.raises(ValueError):
        test_module.load_state_dict(weights)


def test_load_state_dict_nonstrict(test_module: TestModule) -> None:
    weights = {
        "a": Tensor.constant(5),
        "sub.b": Tensor.constant(6),
        "extra": Tensor.constant(7),
    }
    test_module.load_state_dict(weights, strict=False)
    assert test_module.a.item() == 5
    assert test_module.sub.b.item() == 6


def test_load_state_dict_dtype_mismatch() -> None:
    """Test that load_state_dict raises ValueError for dtype mismatch."""

    @module_dataclass
    class SimpleModule(Module[[Tensor], Tensor]):
        weight: Tensor

        def forward(self, x: Tensor) -> Tensor:
            return x + self.weight

    # Create module with float32 weight
    module = SimpleModule(weight=Tensor.zeros([3, 3], dtype=DType.float32))

    # Try to load int32 weights - should fail
    weights = {"weight": Tensor.zeros([3, 3], dtype=DType.int32)}

    with pytest.raises(ValueError, match="not assignable"):
        module.load_state_dict(weights)


def test_load_state_dict_shape_mismatch() -> None:
    """Test that load_state_dict raises ValueError for shape mismatch."""

    @module_dataclass
    class SimpleModule(Module[[Tensor], Tensor]):
        weight: Tensor

        def forward(self, x: Tensor) -> Tensor:
            return x + self.weight

    # Create module with [3, 3] weight
    module = SimpleModule(weight=Tensor.zeros([3, 3], dtype=DType.float32))

    # Try to load [4, 4] weights - should fail
    weights = {"weight": Tensor.zeros([4, 4], dtype=DType.float32)}

    with pytest.raises(ValueError, match="not assignable"):
        module.load_state_dict(weights)


def test_load_state_dict_dtype_and_shape_mismatch() -> None:
    """Test that load_state_dict raises ValueError when both dtype and shape mismatch."""

    @module_dataclass
    class SimpleModule(Module[[Tensor], Tensor]):
        weight: Tensor

        def forward(self, x: Tensor) -> Tensor:
            return x + self.weight

    # Create module with float32 [3, 3] weight
    module = SimpleModule(weight=Tensor.zeros([3, 3], dtype=DType.float32))

    # Try to load int32 [4, 4] weights - should fail
    weights = {"weight": Tensor.zeros([4, 4], dtype=DType.int32)}

    with pytest.raises(ValueError, match="not assignable"):
        module.load_state_dict(weights)


def test_load_state_dict_valid_types() -> None:
    """Test that load_state_dict succeeds when dtype and shape match."""

    @module_dataclass
    class SimpleModule(Module[[Tensor], Tensor]):
        weight: Tensor

        def forward(self, x: Tensor) -> Tensor:
            return x + self.weight

    # Create module with float32 [3, 3] weight initialized to zeros
    module = SimpleModule(weight=Tensor.zeros([3, 3], dtype=DType.float32))

    # Load matching weights with ones - should succeed
    weights = {"weight": Tensor.ones([3, 3], dtype=DType.float32)}
    module.load_state_dict(weights)

    # Verify the weights were loaded (first element should be 1.0, not 0.0)
    assert module.weight[0, 0].item() == 1.0


@pytest.mark.skipif(not accelerator_count(), reason="requires multiple devices")
def test_to(test_module: TestModule) -> None:
    assert all(t.device == Accelerator() for _, t in test_module.parameters)
    module = test_module.to(CPU())
    assert module is test_module
    assert all(t.device == CPU() for _, t in test_module.parameters)


def test_compile(test_module: TestModule) -> None:
    dtype, device = defaults()
    type = TensorType(dtype, ["batch", "n"], device=device)
    compiled = test_module.compile(type)

    input = random.uniform([3, 3])
    result_eager = test_module(input)
    result_compiled = compiled(input)

    assert all((result_eager == result_compiled)._values())


def test_compile_with_weights_shape_mismatch() -> None:
    @module_dataclass
    class SimpleModule(Module[[Tensor], Tensor]):
        weight: Tensor

        def forward(self, x: Tensor) -> Tensor:
            return x + self.weight

    module = SimpleModule(weight=Tensor.zeros([3, 3], dtype=DType.float32))
    dtype, device = defaults()
    type = TensorType(dtype, [3, 3], device=device)
    weights = {
        "weight": Tensor.zeros([4, 4], dtype=DType.float32),
    }

    with pytest.raises(ValueError, match="not assignable"):
        module.compile(type, weights=weights)


def test_compile_with_weights_dtype_mismatch() -> None:
    @module_dataclass
    class SimpleModule(Module[[Tensor], Tensor]):
        weight: Tensor

        def forward(self, x: Tensor) -> Tensor:
            return x + self.weight

    module = SimpleModule(weight=Tensor.zeros([3, 3], dtype=DType.float32))
    dtype, device = defaults()
    type = TensorType(dtype, [3, 3], device=device)
    weights = {
        "weight": Tensor.zeros([3, 3], dtype=DType.int32),
    }

    with pytest.raises(ValueError, match="not assignable"):
        module.compile(type, weights=weights)


def test_compile_with_weights_missing_parameter_raises() -> None:
    @module_dataclass
    class SimpleModule(Module[[Tensor], Tensor]):
        weight: Tensor

        def forward(self, x: Tensor) -> Tensor:
            return x + self.weight

    module = SimpleModule(weight=Tensor.zeros([3, 3], dtype=DType.float32))
    dtype, device = defaults()
    type = TensorType(dtype, [3, 3], device=device)

    with pytest.raises(KeyError, match="is missing"):
        module.compile(type, weights={})


def test_compile_with_weights(lazy_test_module: TestModule) -> None:
    test_module = lazy_test_module
    dtype, device = defaults()
    type = TensorType(dtype, ["batch", "n"], device=device)

    parameters = weakref.WeakValueDictionary(test_module.parameters)

    weights = {
        name: driver.Buffer.zeros(
            [int(d) for d in param.shape], param.dtype, param.device
        )
        for name, param in test_module.parameters
    }

    assert not any(param.real for param in parameters.values())
    assert not any(param.real for _, param in test_module.parameters)

    compiled = test_module.compile(type, weights=weights)

    assert not any(param.real for param in parameters.values())
    assert not any(param.real for _, param in test_module.parameters)

    input = driver.Buffer.zeros([3, 3], dtype, device)
    _ = compiled(input)

    assert not any(param.real for param in parameters.values())
    assert not any(param.real for _, param in test_module.parameters)


def test_compile_with_weights_never_realized(
    lazy_test_module: TestModule,
) -> None:
    test_module = lazy_test_module
    dtype, device = defaults()
    type = TensorType(dtype, ["batch", "n"], device=device)

    parameters = weakref.WeakValueDictionary(test_module.parameters)

    weights = {
        name: Tensor.zeros_like(param.type)
        for name, param in test_module.parameters
    }

    assert not any(param.real for param in parameters.values())
    assert not any(param.real for _, param in test_module.parameters)

    compiled = test_module.compile(type, weights=weights)

    assert not any(param.real for param in parameters.values())
    assert not any(param.real for _, param in test_module.parameters)

    input = random.uniform([3, 3])
    _ = compiled(input)

    assert not any(param.real for param in parameters.values())
    assert not any(param.real for _, param in test_module.parameters)
