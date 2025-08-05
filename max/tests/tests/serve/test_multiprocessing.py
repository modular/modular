# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from multiprocessing.reduction import ForkingPickler

import numpy
from max.interfaces import TextGenerationRequest


def test_reductions() -> None:
    # No extra reductions to register at the moment.

    request = TextGenerationRequest(
        request_id="0", prompt="test", model_name="test"
    )
    context = {
        "0": numpy.ones((3, 3), dtype=numpy.float32),
    }
    for obj in (request, context):
        assert ForkingPickler.dumps(obj)
