# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# TODO(asan): Timing out in ASAN. Fix.
# UNSUPPORTED: asan

# RUN: echo '[{"id": "%S/test_unknown.test::unknown_test()"}]' > %t
# RUN: not mojo-test-executor --pretty %t  | FileCheck %s --check-prefix=CHECK-UNKNOWN

# RUN: echo '[{"id": "%s::test_fn_assert_failure()"}]' > %t
# RUN: mojo-test-executor --pretty %t | FileCheck %s --check-prefix=CHECK-UNIT1

# RUN: echo '[{"id": "%s::test_def_assert_failure()"}]' > %t
# RUN: mojo-test-executor --pretty %t | FileCheck %s --check-prefix=CHECK-UNIT2

# RUN: echo '[{"id": "%s@doc_test_failure_first_cell().__doc__::0",' > %t
# RUN: echo ' "location": {"endColumn": 1, "endLine": 85,' >> %t
# RUN: echo ' "startColumn": 1, "startLine": 82}},' >> %t
# RUN: echo '{"id": "%s@doc_test_failure_first_cell().__doc__::1",' >> %t
# RUN: echo ' "location": {"endColumn": 1, "endLine": 89,' >> %t
# RUN: echo ' "startColumn": 1, "startLine": 87}}]' >> %t
# RUN: mojo-test-executor --pretty %t | FileCheck %s --check-prefix=CHECK-DOC-FIRST-CELL

# RUN: echo '[{"id": "%s@doc_test_failure_second_cell().__doc__::0",' > %t
# RUN: echo ' "location": {"endColumn": 1, "endLine": 109,' >> %t
# RUN: echo ' "startColumn": 1, "startLine": 107}},' >> %t
# RUN: echo '{"id": "%s@doc_test_failure_second_cell().__doc__::1",' >> %t
# RUN: echo ' "location": {"endColumn": 1, "endLine": 114,' >> %t
# RUN: echo ' "startColumn": 1, "startLine": 111}}]' >> %t
# RUN: mojo-test-executor --pretty %t | FileCheck %s --check-prefix=CHECK-DOC-SECOND-CELL

from testing import assert_true

# CHECK-UNKNOWN: execution/result
# CHECK-UNKNOWN:  "error": "id does not correspond to a valid mojo source file",
# CHECK-UNKNOWN:  "kind": "initializationError",
# CHECK-UNKNOWN:  "testID": "{{.*}}test_unknown.test::unknown_test()"

# CHECK-UNIT1: execution/result
# CHECK-UNIT1:  "error": "Unhandled exception caught during execution",
# CHECK-UNIT1:  "kind": "executionError",
# CHECK-UNIT1:  "stdErr": "{{.*}}error: execution exited with a non-zero result: 1\n",
# CHECK-UNIT1:  "stdOut": {{.*}} AssertionError: condition was unexpectedly False
# CHECK-UNIT1:  "testID": "{{.*}}test_execution.mojo::test_{{fn|def}}_assert_failure()"


fn test_fn_assert_failure() raises:
    assert_true(False)
    return


# CHECK-UNIT2: execution/result
# CHECK-UNIT2:  "error": "Unhandled exception caught during execution",
# CHECK-UNIT2:  "kind": "executionError",
# CHECK-UNIT2:  "stdErr": "{{.*}}error: execution exited with a non-zero result: 1\n",
# CHECK-UNIT2:  "stdOut": {{.*}} AssertionError: condition was unexpectedly False
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
