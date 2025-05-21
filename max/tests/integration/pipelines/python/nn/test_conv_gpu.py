# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from max.engine.api import InferenceSession
from shared_conv_impl import conv1d_impl, conv3d_impl
from test_common.numerics import pytorch_disable_tf32_dtype


@pytorch_disable_tf32_dtype
def test_conv3d_gpu(gpu_session: InferenceSession) -> None:
    conv3d_impl(gpu_session)


@pytorch_disable_tf32_dtype
def test_conv1d_gpu(gpu_session: InferenceSession) -> None:
    conv1d_impl(gpu_session)
