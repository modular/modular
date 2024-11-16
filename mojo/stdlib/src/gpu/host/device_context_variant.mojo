# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from gpu.host.device_context_v2 import (
    DeviceBufferV2,
    DeviceContextV2,
    DeviceFunctionV2,
    DeviceStreamV2,
)

# DeviceContextVariant is not being used any longer. Instead we just define
# aliases to Device*V2 here, with the goal of removing this file entirely.

alias DeviceFunction = DeviceFunctionV2
alias DeviceBuffer = DeviceBufferV2
alias DeviceContext = DeviceContextV2
alias DeviceStream = DeviceStreamV2
