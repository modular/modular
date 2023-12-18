# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# RUN: %mojo -I %engine_pkg_dir %s | FileCheck %s

from aiengine import (
    get_version,
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


fn main() raises:
    test_engine_version()
    test_session()
