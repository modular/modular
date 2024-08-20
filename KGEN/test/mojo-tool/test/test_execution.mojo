# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# TODO(asan): Timing out in ASAN. Fix.
# UNSUPPORTED: asan

# RUN: not mojo test -I %S/inputs -D TEST_PASS "%s" | FileCheck %s
# RUN: not mojo test -I %S/inputs "%s@doc_test_failure_first_cell().__doc__" | FileCheck %s --check-prefix=CHECK-DOC
# RUN: not mojo test -I %S/inputs "%s@doc_test_failure_first_cell().__doc__::1" | FileCheck %s --check-prefix=CHECK-DOC
# RUN: not mojo test -I %S/inputs "%s::test_unit\2Efailure()" | FileCheck %s --check-prefix=CHECK-UNIT
# RUN: not mojo test -I %S/inputs --diagnostic-format json "%s@doc_test_failure_first_cell().__doc__" | FileCheck %s --check-prefix=CHECK-DOC-JSON

from imported_module import returns_false
from testing import assert_true
from sys import is_defined

# CHECK: Testing Time: {{.*}}s
# CHECK: Total Discovered Tests: 9
# CHECK: Passed : 5
# CHECK: Failed : 3
# CHECK: Skipped: 1

# CHECK: Failure: '{{.*}}test_execution.mojo@doc_test_failure_second_cell().__doc__::1'
# CHECK: Unhandled exception caught during execution
# CHECK: Error: {{.*}} AssertionError: condition was unexpectedly False

# CHECK: Failure: '{{.*}}test_execution.mojo@doc_test_failure_first_cell().__doc__::0'
# CHECK: Unhandled exception caught during execution
# CHECK: Error: {{.*}} AssertionError: condition was unexpectedly False

# CHECK: Failure: '{{.*}}test_execution.mojo::test_unit\2Efailure()'
# CHECK: Unhandled exception caught during execution
# CHECK: {{.*}}test_execution.mojo:61:16: AssertionError: condition was unexpectedly False

# CHECK-DOC: Total Discovered Tests: 2
# CHECK-DOC: Passed : 0
# CHECK-DOC: Failed : 1
# CHECK-DOC: Skipped: 1

# CHECK-UNIT: Total Discovered Tests: 1
# CHECK-UNIT: Passed : 0
# CHECK-UNIT: Failed : 1
# CHECK-UNIT: Skipped: 0

# CHECK-DOC-JSON:   "children": [
# CHECK-DOC-JSON:       "error": "Unhandled exception caught during execution",
# CHECK-DOC-JSON:       "kind": "executionError",
# CHECK-DOC-JSON:       "stdOut": "Error: {{.*}} AssertionError: condition was unexpectedly False\r\n",
# CHECK-DOC-JSON:       "testID": "{{.*}}test_execution.mojo@doc_test_failure_first_cell().__doc__::0"

# CHECK-DOC-JSON:       "kind": "skipped",
# CHECK-DOC-JSON:       "testID": "{{.*}}test_execution.mojo@doc_test_failure_first_cell().__doc__::1"
# CHECK-DOC-JSON:   "kind": "executionError",
# CHECK-DOC-JSON:   "testID": "{{.*}}test_execution.mojo@doc_test_failure_first_cell().__doc__"


fn `test_unit.failure`() raises:
    assert_true(returns_false())
    return


fn test_unit_pass():
    return


fn doc_test_failure_first_cell():
    """This is a doc string.

    ```mojo
    from testing import assert_true
    from imported_module import returns_false
    assert_true(returns_false())
    ```

    ```mojo
    print("hello")
    ```
    """
    return


fn doc_test_failure_second_cell():
    """This is a doc string.

    ```mojo
    var value = False
    ```

    ```mojo
    from testing import assert_true
    assert_true(value)
    ```
    """
    return


fn doc_test_pass():
    """This is a doc string.

    ```mojo
    var value = True
    ```

    ```mojo
    from testing import assert_true
    assert_true(value)
    ```
    """
    return


fn test_def() raises:
    assert_true(is_defined["TEST_PASS"]())
