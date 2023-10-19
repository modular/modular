# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# RUN: %mojo -I %engine_pkg_dir %s | FileCheck %s

from aiengine import (
    get_version,
    EngineDType,
    InferenceSession,
)


fn test_engine_version() raises:
    # CHECK: test_version
    print("====test_version")

    # CHECK: Version: {{.*}}
    print("Version:", get_version())


fn test_session() raises:
    # CHECK: test_session
    print("====test_session")

    let session = InferenceSession()


fn test_dtype() raises:
    # CHECK: test_dtype
    print("====test_dtype")

    # CHECK: True
    print(EngineDType(DType.int8) == EngineDType.int8)

    # CHECK: True
    print(EngineDType(DType.int8).to_dtype() == DType.int8)


fn main() raises:
    test_engine_version()
    test_session()
    test_dtype()
