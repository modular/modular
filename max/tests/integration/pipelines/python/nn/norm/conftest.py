# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pytest
from max.driver import Accelerator
from max.engine import InferenceSession


@pytest.fixture(scope="session")
def session() -> InferenceSession:
    return InferenceSession()


@pytest.fixture(scope="module")
def gpu_session() -> InferenceSession:
    return InferenceSession(devices=[Accelerator()])
