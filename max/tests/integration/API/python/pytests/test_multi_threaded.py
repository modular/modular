# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the modular.engine on multi-threaded scenarios"""

import concurrent.futures
import multiprocessing
from pathlib import Path

import numpy as np
import pytest
from test_utils import generate_test_inputs

import modular.engine as me


@pytest.mark.parametrize(
    "model_path_fixture",
    ["tf_basic_mlp", "tf_dlrm", "tf_bert"],
)
def test_multithread_execute_tensorflow(
    model_path_fixture: str, request
) -> None:
    model_path: Path = request.getfixturevalue(model_path_fixture)
    # number of threads our runtime uses is strictly orthogonal to multi-threading below
    # this is just to be explicit
    session = me.InferenceSession(num_threads=4)
    model: me.Model = session.load(model_path)
    assert model is not None

    num_workers = 4
    args = zip(*(generate_test_inputs(model) for _ in range(num_workers)))

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_workers
    ) as executor:
        executor.map(model.execute, *args)


@pytest.mark.parametrize(
    "model_path_fixture",
    ["tf_basic_mlp", "tf_dlrm", "tf_bert"],
)
def test_multiprocess_execute_tensorflow(
    model_path_fixture: str, request
) -> None:
    model_path: Path = request.getfixturevalue(model_path_fixture)
    session = me.InferenceSession(num_threads=4)
    model: me.Model = session.load(model_path)
    assert model is not None

    num_workers = 4
    args = zip(*(generate_test_inputs(model) for _ in range(num_workers)))

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_workers
    ) as executor:
        executor.map(model.execute, *args)
