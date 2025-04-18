# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the Python weight loading interface."""

import platform

import numpy as np
import torch
from max.driver import CPU, Tensor, accelerator_count
from max.dtype import DType
from max.graph import DeviceRef, Graph, Weight
from max.graph.weights import GGUFWeights, PytorchWeights, SafetensorWeights


def test_weight(session) -> None:
    """Tests adding an external weight to a graph."""
    with Graph("graph_with_weights") as graph:
        weight_shape = [5, 10]
        weight = np.random.uniform(1, 100, size=weight_shape).astype(np.int64)
        w = Weight(
            "random_weight",
            dtype=DType.int64,
            shape=weight_shape,
            device=DeviceRef.CPU(),
        )
        device_ref = (
            DeviceRef.CPU() if accelerator_count() == 0 else DeviceRef.GPU()
        )
        out = w.to(device_ref) * 2
        graph.output(w.to(device_ref), out)
        compiled = session.load(
            graph, weights_registry={"random_weight": weight}
        )
        output = compiled.execute()
        output0 = output[0]
        output1 = output[1]
        assert isinstance(output0, Tensor)
        assert isinstance(output1, Tensor)
        np.testing.assert_array_equal(weight, output0.to_numpy())
        np.testing.assert_array_equal(weight * 2, output1.to_numpy())


def test_weight_offset(session) -> None:
    """Tests adding an external weight to a graph."""
    with Graph("graph_with_offset_weights") as graph:
        weight_shape = [5, 10]
        weight = np.random.uniform(1, 100, size=weight_shape).astype(np.int64)
        w = Weight(
            "random_weight",
            dtype=DType.int64,
            shape=weight_shape,
            device=DeviceRef.CPU(),
        )
        device_ref = (
            DeviceRef.CPU() if accelerator_count() == 0 else DeviceRef.GPU()
        )
        graph.output(w.to(device_ref))
        compiled = session.load(
            graph, weights_registry={"random_weight": weight}
        )
        output = compiled.execute()
        output0 = output[0]
        assert isinstance(output0, Tensor)
        np.testing.assert_array_equal(weight, output0.to_numpy())


def _test_data():
    # supported on all platforms
    data = {
        "a": np.arange(10, dtype=np.int32).reshape(5, 2),
        "b": np.full((1, 2, 3), 3.5, dtype=np.float64),
        "c": np.array(5432.1, dtype=np.float32),
        "fancy/name": np.array([1, 2, 3], dtype=np.int64),
    }
    # supported on gpu
    if accelerator_count() > 0:
        data["bf16"] = torch.tensor([123, 45], dtype=torch.bfloat16)
        data["float8_e4m3fn"] = torch.tensor(
            [11.0, 250.0], dtype=torch.float8_e4m3fn
        )
        data["float8_e5m2"] = torch.tensor(
            [13.0, 223.0], dtype=torch.float8_e5m2
        )
        return data

    # supported if not arm
    if platform.machine() not in ["arm64", "aarch64"]:
        data["bf16"] = torch.tensor([123, 45], dtype=torch.bfloat16)

    return data


def test_load_pytorch(session, graph_testdata) -> None:
    """Tests adding an external weight to a graph."""
    expected_dict = _test_data()
    # pytorch weights file does not currently contain float8 weights
    expected_dict.pop("float8_e4m3fn", None)
    expected_dict.pop("float8_e5m2", None)
    flat_keys = list(expected_dict.keys())
    expected = [expected_dict[k] for k in flat_keys]

    weights = PytorchWeights(graph_testdata / "example_data.pt")
    with Graph("graph_with_pt_weights") as graph:
        loaded = {k: graph.add_weight(w.allocate()) for k, w in weights.items()}
        device_ref = (
            DeviceRef.CPU() if accelerator_count() == 0 else DeviceRef.GPU()
        )
        graph.output(*[loaded[k].to(device_ref) for k in flat_keys])
        compiled = session.load(
            graph, weights_registry=weights.allocated_weights
        )
        output = compiled.execute()

        assert len(expected) == len(output)
        for n, expected_tensor in enumerate(expected):
            if flat_keys[n] == "bf16":
                assert torch.equal(
                    expected_tensor, torch.from_dlpack(output[n].to(CPU()))
                )
            elif any(
                flat_keys[n].endswith(suffix)
                for suffix in ["float8_e4m3fn", "float8_e5m2"]
            ):
                assert torch.equal(
                    expected_tensor.view(torch.uint8),
                    torch.from_dlpack(output[n].to(CPU()).view(DType.uint8)),
                )
            else:
                np.testing.assert_array_equal(
                    expected_tensor, output[n].to_numpy()
                )


def test_load_gguf(session, graph_testdata) -> None:
    """Tests adding an external weight to a graph."""
    expected_dict = _test_data()
    # gguf weights file does not currently contain float8 weights
    expected_dict.pop("float8_e4m3fn", None)
    expected_dict.pop("float8_e5m2", None)
    expected_dict["quantized"] = np.arange(0, 288, dtype=np.uint8).reshape(
        2, 144
    )

    flat_keys = list(expected_dict.keys())
    expected = [expected_dict[k] for k in flat_keys]

    weights = GGUFWeights(graph_testdata / "example_data.gguf")
    with Graph("graph_with_gguf_weights") as graph:
        loaded = {k: graph.add_weight(w.allocate()) for k, w in weights.items()}
        device_ref = (
            DeviceRef.CPU() if accelerator_count() == 0 else DeviceRef.GPU()
        )
        graph.output(*[loaded[k].to(device_ref) for k in flat_keys])
        compiled = session.load(
            graph, weights_registry=weights.allocated_weights
        )
        output = compiled.execute()

        assert len(expected) == len(output)
        for n, expected_tensor in enumerate(expected):
            if flat_keys[n] == "bf16":
                assert torch.equal(
                    expected_tensor, torch.from_dlpack(output[n].to(CPU()))
                )
            elif any(
                flat_keys[n].endswith(suffix)
                for suffix in ["float8_e4m3fn", "float8_e5m2"]
            ):
                assert torch.equal(
                    expected_tensor.view(torch.uint8),
                    torch.from_dlpack(output[n].to(CPU()).view(DType.uint8)),
                )
            else:
                np.testing.assert_array_equal(
                    expected_tensor, output[n].to_numpy()
                )


def test_load_safetensors(session, graph_testdata) -> None:
    """Tests adding an external weight to a graph."""
    expected_base_dict = _test_data()
    expected_dict = {
        f"{i}.{k}": v
        for k, v in expected_base_dict.items()
        for i in range(1, 3)
    }
    flat_keys = list(expected_dict.keys())
    expected = [expected_dict[k] for k in flat_keys]

    weights = SafetensorWeights(
        [graph_testdata / f"example_data_{i}.safetensors" for i in range(1, 3)]
    )
    with Graph("graph_with_pt_weights") as graph:
        loaded = {k: graph.add_weight(w.allocate()) for k, w in weights.items()}
        device_ref = (
            DeviceRef.CPU() if accelerator_count() == 0 else DeviceRef.GPU()
        )
        graph.output(*[loaded[k].to(device_ref) for k in flat_keys])
        compiled = session.load(
            graph,
            weights_registry={
                k: Tensor.from_numpy(v)
                for k, v in weights.allocated_weights.items()
            },
        )

        output = compiled.execute()
        assert len(expected) == len(output)
        for n, expected_tensor in enumerate(expected):
            if flat_keys[n].endswith("bf16"):
                assert torch.equal(
                    expected_tensor, torch.from_dlpack(output[n].to(CPU()))
                )
            elif any(
                flat_keys[n].endswith(suffix)
                for suffix in ["float8_e4m3fn", "float8_e5m2"]
            ):
                assert torch.equal(
                    expected_tensor.view(torch.uint8),
                    torch.from_dlpack(output[n].to(CPU()).view(DType.uint8)),
                )
            else:
                np.testing.assert_array_equal(
                    expected_tensor, output[n].to_numpy()
                )
