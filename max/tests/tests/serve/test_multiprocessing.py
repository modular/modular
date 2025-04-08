# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from multiprocessing.reduction import ForkingPickler

import numpy
from max.pipelines.core import TokenGeneratorRequest


def test_reductions():
    # No extra reductions to register at the moment.

    request = TokenGeneratorRequest(
        id="0", index=0, prompt="test", model_name="test"
    )
    context = {
        "0": numpy.ones((3, 3), dtype=numpy.float32),
    }
    for obj in (request, context):
        assert ForkingPickler.dumps(obj)
