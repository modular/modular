# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import platform

if config.root.host_os == "Windows":
    # TODO(#13522): LLDB currently isn't built on windows.
    config.unsupported = True
elif config.root.host_os == "Darwin" and platform.processor() == "x86_64":
    # TODO(#20407): LLDB and Jupyter tests fail on macOS x86_64.
    config.unsupported = True
