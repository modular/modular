# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# TODO(#13522): LLDB currently isn't built on windows.
if config.root.host_os == "Windows":
    config.unsupported = True
