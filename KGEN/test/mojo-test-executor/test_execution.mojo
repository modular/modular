# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# TODO(asan): Timing out in ASAN. Fix.
# UNSUPPORTED: asan

# RUN: not mojo-test-executor --pretty "%s::unknown_test()" | FileCheck %s --check-prefix=CHECK-UNKNOWN
# RUN: mojo-test-executor --pretty "%s::test_fn_assert_failure()" | FileCheck %s --check-prefix=CHECK-UNIT1
# RUN: mojo-test-executor --pretty "%s::test_def_assert_failure()" | FileCheck %s --check-prefix=CHECK-UNIT2
# RUN: mojo-test-executor --pretty "%s@doc_test_failure_first_cell().__doc__::1" | FileCheck %s --check-prefix=CHECK-DOC-FIRST-CELL
# RUN: mojo-test-executor --pretty "%s@doc_test_failure_first_cell().__doc__::0" | FileCheck %s --check-prefix=CHECK-DOC-ONLY-FIRST-CELL
# RUN: mojo-test-executor --pretty "%s@doc_test_failure_second_cell().__doc__::1" | FileCheck %s --check-prefix=CHECK-DOC-SECOND-CELL

from testing import assert_true

# CHECK-UNKNOWN: execution/result
# CHECK-UNKNOWN:  "error": "id does not reference a valid test",
# CHECK-UNKNOWN:  "kind": "initializationError",
# CHECK-UNKNOWN:  "testID": "{{.*}}test_execution.mojo::unknown_test()"

# CHECK-UNIT1: execution/result
# CHECK-UNIT1:  "error": "Unhandled exception caught during execution",
# CHECK-UNIT1:  "kind": "executionError",
# CHECK-UNIT1:  "stdOut": {{.*}}Error: At {{.*}}test_execution.mojo:32:16: AssertionError: condition was unexpectedly False\r\n",
# CHECK-UNIT1:  "testID": "{{.*}}test_execution.mojo::test_{{fn|def}}_assert_failure()"


fn test_fn_assert_failure() raises:
    assert_true(False)
    return


# CHECK-UNIT2: execution/result
# CHECK-UNIT2:  "error": "Unhandled exception caught during execution",
# CHECK-UNIT2:  "kind": "executionError",
# CHECK-UNIT2:  "stdOut": {{.*}}Error: At {{.*}}test_execution.mojo:44:16: AssertionError: condition was unexpectedly False\r\n",
# CHECK-UNIT2:  "testID": "{{.*}}test_execution.mojo::test_{{fn|def}}_assert_failure()"


def test_def_assert_failure():
    assert_true(False)
    return


# CHECK-DOC-FIRST-CELL: execution/result
# CHECK-DOC-FIRST-CELL:     "error": "Unhandled exception caught during execution",
# CHECK-DOC-FIRST-CELL:     "kind": "executionError",
# CHECK-DOC-FIRST-CELL:     "stdOut": {{.*}}Error: {{.*}} AssertionError: condition was unexpectedly False
# CHECK-DOC-FIRST-CELL:     "testID": "{{.*}}test_execution.mojo@doc_test_failure_first_cell().__doc__::0"
# CHECK-DOC-FIRST-CELL: execution/result
# CHECK-DOC-FIRST-CELL:     "error": "",
# CHECK-DOC-FIRST-CELL:     "kind": "skipped",
# CHECK-DOC-FIRST-CELL:     "testID": "{{.*}}test_execution.mojo@doc_test_failure_first_cell().__doc__::1"

# CHECK-DOC-ONLY-FIRST-CELL: execution/result
# CHECK-DOC-ONLY-FIRST-CELL:     "kind": "executionError",
# CHECK-DOC-ONLY-FIRST-CELL:     "testID": "{{.*}}test_execution.mojo@doc_test_failure_first_cell().__doc__::0"
# CHECK-DOC-ONLY-FIRST-CELL-NOT: "testID": "{{.*}}test_execution.mojo@doc_test_failure_first_cell().__doc__::1"


fn doc_test_failure_first_cell():
    """This is a doc string.

    ```mojo
    from testing import assert_true
    assert_true(False)
    ```

    ```mojo
    print("hello")
    ```
    """
    return


# CHECK-DOC-SECOND-CELL: execution/result
# CHECK-DOC-SECOND-CELL:     "kind": "success",
# CHECK-DOC-SECOND-CELL:     "testID": "{{.*}}test_execution.mojo@doc_test_failure_second_cell().__doc__::0"
# CHECK-DOC-SECOND-CELL: execution/result
# CHECK-DOC-SECOND-CELL:     "error": "Unhandled exception caught during execution",
# CHECK-DOC-SECOND-CELL:     "kind": "executionError",
# CHECK-DOC-SECOND-CELL:     "stdOut": {{.*}}Error: {{.*}} AssertionError: condition was unexpectedly False\r\n",
# CHECK-DOC-SECOND-CELL:     "testID": "{{.*}}test_execution.mojo@doc_test_failure_second_cell().__doc__::1"


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
