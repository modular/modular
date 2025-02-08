# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from pathlib import Path

import numpy as np
import pytest
import torch
from max.driver import CPU, Accelerator, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph
from max.graph.weights import SafetensorWeights


def _test_data():
    return {
        "a": np.arange(10, dtype=np.int32).reshape(5, 2),
        "b": np.full((1, 2, 3), 3.5, dtype=np.float64),
        "c": np.array(5432.1, dtype=np.float32),
        "fancy/name": np.array([1, 2, 3], dtype=np.int64),
        # This is actually saved as bf16 in gen_external_checkpoints.
        "bf16": torch.tensor([123, 45], dtype=torch.bfloat16),
        "float8_e4m3fn": torch.tensor([11.0, 250.0], dtype=torch.float8_e4m3fn),
        "float8_e5m2": torch.tensor([13.0, 223.0], dtype=torch.float8_e5m2),
    }


@pytest.mark.skip(reason="Skipping test see GENAI-63")
def test_load_safetensors(
    gpu_session: InferenceSession, graph_testdata: Path
) -> None:
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
        loaded = {
            k: graph.add_weight(w.allocate(), device=DeviceRef.CPU())
            for k, w in weights.items()
        }
        graph.output(*[loaded[k].to(DeviceRef.GPU()) for k in flat_keys])
        compiled = gpu_session.load(
            graph,
            weights_registry={
                k: Tensor.from_numpy(v).to(Accelerator())
                for k, v in weights.allocated_weights.items()
            },
        )

        output = compiled.execute()
        assert len(expected) == len(output)
        for n, expected in enumerate(expected):
            if flat_keys[n].endswith("bf16"):
                assert torch.equal(
                    expected, torch.from_dlpack(output[n].to(CPU()))
                )
            elif any(
                flat_keys[n].endswith(suffix)
                for suffix in ["float8_e4m3fn", "float8_e5m2"]
            ):
                assert torch.equal(
                    expected.view(torch.uint8),
                    torch.from_dlpack(output[n].to(CPU()).view(DType.uint8)),
                )
            else:
                np.testing.assert_array_equal(expected, output[n].to_numpy())
