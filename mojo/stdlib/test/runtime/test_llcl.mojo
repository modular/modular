# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen %s -execute -func='$test_llcl::main():index()' -I %stdlibdir | FileCheck %s

from IO import print
from Int import Int
from LLCL import Runtime

# CHECK-LABEL: test_llcl
fn test_llcl():
    print("== test_llcl\n")

    async fn test_llcl_add[lhs: Int](rhs: Int) -> Int:
        return lhs + rhs

    async fn test_llcl_add_two_of_them(a: Int, b: Int) -> Int:
        let rt = Runtime.get_current()
        let t0 = rt.run_task[Int](test_llcl_add[1](a))
        let t1 = rt.run_task[Int](test_llcl_add[2](b))
        let result = await t0 + await t1
        t0.__del__()
        t1.__del__()
        return result

    let rt = Runtime(4)
    let task = rt.init_and_run[Int](test_llcl_add_two_of_them(10, 20))
    # CHECK: 33
    print(task.wait())
    task.__del__()
    rt.__del__()


@export
fn main() -> __mlir_type.index:
    test_llcl()
    return 0
