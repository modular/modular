# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s
from algorithm.sort import sort, partition

from utils.vector import DynamicVector


# CHECK-LABEL: test_sort0
fn test_sort0():
    print("== test_sort0")

    var vector = DynamicVector[Int]()

    sort(vector)

    vector._del_old()


# CHECK-LABEL: test_sort2
fn test_sort2():
    print("== test_sort2")

    alias length = 2
    var vector = DynamicVector[Int]()

    vector.push_back(-1)
    vector.push_back(0)

    sort(vector)

    # CHECK: -1
    # CHECK: 0
    for i in range(length):
        print(vector[i])

    vector[0] = 2
    vector[1] = -2

    sort(vector)

    # CHECK: -2
    # CHECK: 2
    for i in range(length):
        print(vector[i])

    vector._del_old()


# CHECK-LABEL: test_sort3
fn test_sort3():
    print("== test_sort3")

    alias length = 3
    var vector = DynamicVector[Int]()

    vector.push_back(-1)
    vector.push_back(0)
    vector.push_back(1)

    sort(vector)

    # CHECK: -1
    # CHECK: 0
    # CHECK: 1
    for i in range(length):
        print(vector[i])

    vector[0] = 2
    vector[1] = -2
    vector[2] = 0

    sort(vector)

    # CHECK: -2
    # CHECK: 0
    # CHECK: 2
    for i in range(length):
        print(vector[i])

    vector._del_old()


# CHECK-LABEL: test_sort4
fn test_sort4():
    print("== test_sort4")

    alias length = 4
    var vector = DynamicVector[Int]()

    vector.push_back(-1)
    vector.push_back(0)
    vector.push_back(1)
    vector.push_back(2)

    sort(vector)

    # CHECK: -1
    # CHECK: 0
    # CHECK: 1
    # CHECK: 2
    for i in range(length):
        print(vector[i])

    vector[0] = 2
    vector[1] = -2
    vector[2] = 0
    vector[3] = -4

    sort(vector)

    # CHECK: -4
    # CHECK: -2
    # CHECK: 0
    # CHECK: 2
    for i in range(length):
        print(vector[i])

    vector._del_old()


# CHECK-LABEL: test_sort5
fn test_sort5():
    print("== test_sort5")

    alias length = 5
    var vector = DynamicVector[Int]()

    for i in range(5):
        vector.push_back(i)

    sort(vector)

    # CHECK: 0
    # CHECK: 1
    # CHECK: 2
    # CHECK: 3
    # CHECK: 4
    for i in range(length):
        print(vector[i])

    vector[0] = 2
    vector[1] = -2
    vector[2] = 0
    vector[3] = -4
    vector[4] = 1

    sort(vector)

    # CHECK: -4
    # CHECK: -2
    # CHECK: 0
    # CHECK: 1
    # CHECK: 2
    for i in range(length):
        print(vector[i])

    vector._del_old()


# CHECK-LABEL: test_sort_reverse
fn test_sort_reverse():
    print("== test_sort_reverse")

    alias length = 5
    var vector = DynamicVector[Int](length)

    for i in range(length):
        vector.push_back(length - i - 1)

    sort(vector)

    # CHECK: 0
    # CHECK: 1
    # CHECK: 2
    # CHECK: 3
    # CHECK: 4
    for i in range(length):
        print(vector[i])

    vector._del_old()


# CHECK-LABEL: test_sort_semi_random
fn test_sort_semi_random():
    print("== test_sort_semi_random")

    alias length = 8
    var vector = DynamicVector[Int](length)

    for i in range(length):
        if i % 2:
            vector.push_back(-i)
        else:
            vector.push_back(i)

    sort(vector)

    # CHECK: 7
    # CHECK: 5
    # CHECK: 3
    # CHECK: 1
    # CHECK: 0
    # CHECK: 2
    # CHECK: 4
    # CHECK: 6
    for i in range(length):
        print(vector[i])

    vector._del_old()


# CHECK-LABEL: test_sort9
fn test_sort9():
    print("== test_sort9")

    alias length = 9
    var vector = DynamicVector[Int](length)

    for i in range(length):
        vector.push_back(length - i - 1)

    sort(vector)

    # CHECK: 0
    # CHECK: 1
    # CHECK: 2
    # CHECK: 3
    # CHECK: 4
    # CHECK: 5
    # CHECK: 6
    # CHECK: 7
    # CHECK: 8
    for i in range(length):
        print(vector[i])

    vector._del_old()


# CHECK-LABEL: test_sort103
fn test_sort103():
    print("== test_sort103")

    alias length = 103
    var vector = DynamicVector[Int](length)

    for i in range(length):
        vector.push_back(length - i - 1)

    sort(vector)

    # CHECK-NOT: unsorted
    for i in range(1, length):
        if vector[i - 1] > vector[i]:
            print("error: unsorted")

    vector._del_old()


# CHECK-LABEL: test_sort_any_103
fn test_sort_any_103():
    print("== test_sort_any_103")

    alias length = 103
    var vector = DynamicVector[Float32](length)

    for i in range(length):
        vector.push_back(length - i - 1)

    sort[DType.float32](vector)

    # CHECK-NOT: unsorted
    for i in range(1, length):
        if vector[i - 1] > vector[i]:
            print("error: unsorted")

    vector._del_old()


fn test_partition_top_k(length: Int, k: Int):
    print_no_newline("== test_partition_top_k_")
    print_no_newline(length)
    print_no_newline("_")
    print_no_newline(k)
    print("")

    var vector = DynamicVector[Float32](length)

    for i in range(0, length):
        vector.push_back(i)

    @parameter
    fn _great_than_equal[type: AnyType](lhs: type, rhs: type) -> Bool:
        return rebind[Float32](lhs) >= rebind[Float32](rhs)

    partition[Float32, _great_than_equal](vector.data, k, vector.__len__())

    for i in range(0, k):
        if vector[i] < length - k:
            print("error: incorrect top-k element", vector[i])
    vector._del_old()


fn main():
    test_sort0()
    test_sort2()
    test_sort3()
    test_sort4()
    test_sort5()
    test_sort_reverse()
    test_sort_semi_random()
    test_sort9()
    test_sort103()
    test_sort_any_103()

    # CHECK-LABEL: test_partition_top_k_7_5
    # CHECK-NOT: incorrect top-k
    test_partition_top_k(7, 5)
    # CHECK-LABEL: test_partition_top_k_11_2
    # CHECK-NOT: incorrect top-k
    test_partition_top_k(11, 2)
    # CHECK-LABEL: test_partition_top_k_4_1
    # CHECK-NOT: incorrect top-k
    test_partition_top_k(4, 1)
