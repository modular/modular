# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Tests for max.nn.Module."""

from __future__ import annotations

import re

import pytest
from max.experimental.tensor import Tensor
from max.nn.module_v3.module import Module, module_dataclass


@module_dataclass
class SubModule(Module):
    b: Tensor
    eps: float = 1e-5

    def __call__(self, x: Tensor) -> Tensor:
        return x + self.b


@module_dataclass
class TestModule(Module):
    a: Tensor
    sub: SubModule

    def __call__(self, x: Tensor) -> Tensor:
        return self.sub(x) + self.a


@module_dataclass
class SuperModule(Module):
    mod: TestModule


@pytest.fixture
def test_module():
    return TestModule(a=Tensor.constant(1), sub=SubModule(b=Tensor.constant(2)))


@pytest.fixture
def super_module(test_module: TestModule):
    return SuperModule(mod=test_module)


def test_module_dataclass():
    @module_dataclass
    class Test(Module):
        a: int
        b: int = 0

    assert repr(Test(2)) == "Test(a=2)"
    assert repr(Test(1, 3)) == "Test(a=1, b=3)"


def test_module_repr(test_module: TestModule):
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


def test_module_custom_repr():
    class Linear(Module):
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


def test_module_decomposition(test_module: TestModule):
    test_module_2 = TestModule(a=Tensor.constant(1), sub=test_module.sub)
    assert test_module_2.sub is test_module.sub
    assert dict(test_module_2.children) == dict(test_module.children)


def test_module_decomposition_call(test_module: TestModule):
    x = Tensor.constant(1)
    assert test_module.sub.b.item() == 2
    assert test_module.sub(x).item() == 3


def test_module_local_parameters(test_module: TestModule):
    assert dict(test_module.local_parameters) == {"a": test_module.a}
    assert dict(test_module.sub.local_parameters) == {"b": test_module.sub.b}


def test_module_parameters(test_module: TestModule):
    assert dict(test_module.parameters) == {
        "a": test_module.a,
        "sub.b": test_module.sub.b,
    }

    assert dict(test_module.sub.parameters) == {"b": test_module.sub.b}


def test_module_children(test_module: TestModule, super_module: SuperModule):
    assert dict(super_module.children) == {"mod": test_module}
    assert dict(test_module.children) == {"sub": test_module.sub}
    assert dict(test_module.sub.children) == {}


def test_module_descendents(test_module: TestModule, super_module: SuperModule):
    assert super_module.mod is test_module
    assert dict(super_module.descendents) == {
        "mod": test_module,
        "mod.sub": test_module.sub,
    }
    assert dict(super_module.mod.descendents) == {"sub": super_module.mod.sub}
    assert dict(test_module.sub.descendents) == {}


def test_apply_to_local_parameters(test_module: TestModule):
    a = test_module.a
    b = test_module.sub.b

    test_module.apply_to_local_parameters(lambda _, t: t + 1)
    # Applied to a
    assert test_module.a.item() == (a + 1).item()
    # Not applied to submodule
    assert test_module.sub.b.item() == b.item()


def test_apply_to_parameters(test_module: TestModule):
    a = test_module.a
    b = test_module.sub.b

    test_module.apply_to_parameters(lambda _, t: t + 1)
    # Applied to a
    assert test_module.a.item() == (a + 1).item()
    # Also applied to submodule
    assert test_module.sub.b.item() == (b + 1).item()


def test_apply_to_parameters__qualified_names(test_module: TestModule):
    names = set()
    expected = dict(test_module.parameters).keys()

    def lookup(name: str, tensor: Tensor):
        names.add(name)
        return tensor

    test_module.apply_to_parameters(lookup)
    assert expected == names


def test_load_state_simple_dict(test_module: TestModule):
    weights = {
        "a": Tensor.constant(5),
        "sub.b": Tensor.constant(6),
    }
    test_module.load_state(weights.__getitem__)
    assert test_module.a.item() == 5
    assert test_module.sub.b.item() == 6


def test_load_state_simple_dict_lookup_failure(test_module: TestModule):
    weights: dict[str, Tensor] = {}
    # No guarantee on the resulting state here!
    with pytest.raises(KeyError):
        test_module.load_state(weights.__getitem__)


def test_load_state_name_remapping(test_module: TestModule):
    def remap_name(name: str):
        name = re.sub(r"\bsub\.", "feed_forward.", name)
        return name

    weights = {
        "a": Tensor.constant(5),
        "feed_forward.b": Tensor.constant(6),
    }

    test_module.load_state(lambda name: weights[remap_name(name)])
    assert test_module.a.item() == 5
    assert test_module.sub.b.item() == 6


def test_load_state_dict(test_module: TestModule):
    weights = {
        "a": Tensor.constant(5),
        "sub.b": Tensor.constant(6),
    }
    test_module.load_state_dict(weights)
    assert test_module.a.item() == 5
    assert test_module.sub.b.item() == 6


def test_load_state_dict_strict(test_module: TestModule):
    weights = {
        "a": Tensor.constant(5),
        "sub.b": Tensor.constant(6),
        "extra": Tensor.constant(7),
    }
    with pytest.raises(ValueError):
        test_module.load_state_dict(weights)


def test_load_state_dict_nonstrict(test_module: TestModule):
    weights = {
        "a": Tensor.constant(5),
        "sub.b": Tensor.constant(6),
        "extra": Tensor.constant(7),
    }
    test_module.load_state_dict(weights, strict=False)
    assert test_module.a.item() == 5
    assert test_module.sub.b.item() == 6
