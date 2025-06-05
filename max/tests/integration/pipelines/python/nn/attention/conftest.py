# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pytest
from max.engine import InferenceSession


@pytest.fixture(scope="session")
def session() -> InferenceSession:
    return InferenceSession()
