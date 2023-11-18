# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s
from algorithm.sort import _small_sort, partition, sort, _quicksort
from random import random_si64, seed
from utils.vector import DynamicVector


# CHECK-LABEL: test_sort_small_3
fn test_sort_small_3():
    print("== test_sort_small_3")
    alias length = 3

    var vector = DynamicVector[Int]()

    vector.push_back(9)
    vector.push_back(1)
    vector.push_back(2)

    @parameter
    fn _less_than_equal[type: AnyRegType](lhs: type, rhs: type) -> Bool:
        return rebind[Int](lhs) <= rebind[Int](rhs)

    _small_sort[length, Int, _less_than_equal](vector.data)

    # CHECK: 1
    # CHECK: 2
    # CHECK: 9
    for i in range(length):
        print(vector[i])

    vector._del_old()


# CHECK-LABEL: test_sort_small_5
fn test_sort_small_5():
    print("== test_sort_small_5")
    alias length = 5

    var vector = DynamicVector[Int]()

    vector.push_back(9)
    vector.push_back(1)
    vector.push_back(2)
    vector.push_back(3)
    vector.push_back(4)

    @parameter
    fn _less_than_equal[type: AnyRegType](lhs: type, rhs: type) -> Bool:
        return rebind[Int](lhs) <= rebind[Int](rhs)

    _small_sort[length, Int, _less_than_equal](vector.data)

    # CHECK: 1
    # CHECK: 2
    # CHECK: 3
    # CHECK: 4
    # CHECK: 9
    for i in range(length):
        print(vector[i])

    vector._del_old()


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


# CHECK-LABEL test_sort3_dupe_elements
fn test_sort3_dupe_elements():
    print("== test_sort3_dupe_elements")

    alias length = 3

    fn test[
        cmp_fn: fn[type: AnyRegType] (type, type) capturing -> Bool,
    ]():
        var vector = DynamicVector[Int](3)
        vector.push_back(5)
        vector.push_back(3)
        vector.push_back(3)

        _quicksort[Int, cmp_fn](vector.data, len(vector))

        # CHECK: 3
        # CHECK: 3
        # CHECK: 5
        for i in range(length):
            print(vector[i])

        vector._del_old()

    @parameter
    fn _lt[type: AnyRegType](lhs: type, rhs: type) -> Bool:
        return rebind[Int](lhs) < rebind[Int](rhs)

    @parameter
    fn _leq[type: AnyRegType](lhs: type, rhs: type) -> Bool:
        return rebind[Int](lhs) <= rebind[Int](rhs)

    test[_lt]()
    test[_leq]()


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


fn test_quick_sort_repeated_val():
    print("==  test_quick_sort_repeated_val")

    alias length = 36
    var vector = DynamicVector[Float32](length)

    for i in range(0, length // 4):
        vector.push_back(i + 1)
        vector.push_back(i + 1)
        vector.push_back(i + 1)
        vector.push_back(i + 1)

    @parameter
    fn _greater_than[type: AnyRegType](lhs: type, rhs: type) -> Bool:
        return rebind[Float32](lhs) > rebind[Float32](rhs)

    _quicksort[Float32, _greater_than](vector.data, len(vector))

    # CHECK: 9.0
    # CHECK: 9.0
    # CHECK: 9.0
    # CHECK: 9.0
    # CHECK: 8.0
    # CHECK: 8.0
    # CHECK: 8.0
    # CHECK: 8.0
    # CHECK: 7.0
    # CHECK: 7.0
    # CHECK: 7.0
    # CHECK: 7.0
    # CHECK: 6.0
    # CHECK: 6.0
    # CHECK: 6.0
    # CHECK: 6.0
    # CHECK: 5.0
    # CHECK: 5.0
    # CHECK: 5.0
    # CHECK: 5.0
    # CHECK: 4.0
    # CHECK: 4.0
    # CHECK: 4.0
    # CHECK: 4.0
    # CHECK: 3.0
    # CHECK: 3.0
    # CHECK: 3.0
    # CHECK: 3.0
    # CHECK: 2.0
    # CHECK: 2.0
    # CHECK: 2.0
    # CHECK: 2.0
    # CHECK: 1.0
    # CHECK: 1.0
    # CHECK: 1.0
    # CHECK: 1.0
    for i in range(0, length):
        print(vector[i])

    @parameter
    fn _less_than[type: AnyRegType](lhs: type, rhs: type) -> Bool:
        return rebind[Float32](lhs) < rebind[Float32](rhs)

    # CHECK: 1.0
    # CHECK: 1.0
    # CHECK: 1.0
    # CHECK: 1.0
    # CHECK: 2.0
    # CHECK: 2.0
    # CHECK: 2.0
    # CHECK: 2.0
    # CHECK: 3.0
    # CHECK: 3.0
    # CHECK: 3.0
    # CHECK: 3.0
    # CHECK: 4.0
    # CHECK: 4.0
    # CHECK: 4.0
    # CHECK: 4.0
    # CHECK: 5.0
    # CHECK: 5.0
    # CHECK: 5.0
    # CHECK: 5.0
    # CHECK: 6.0
    # CHECK: 6.0
    # CHECK: 6.0
    # CHECK: 6.0
    # CHECK: 7.0
    # CHECK: 7.0
    # CHECK: 7.0
    # CHECK: 7.0
    # CHECK: 8.0
    # CHECK: 8.0
    # CHECK: 8.0
    # CHECK: 8.0
    # CHECK: 9.0
    # CHECK: 9.0
    # CHECK: 9.0
    # CHECK: 9.0
    _quicksort[Float32, _less_than](vector.data, len(vector))
    for i in range(0, length):
        print(vector[i])

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
    fn _great_than_equal[type: AnyRegType](lhs: type, rhs: type) -> Bool:
        return rebind[Float32](lhs) >= rebind[Float32](rhs)

    partition[Float32, _great_than_equal](vector.data, k, len(vector))

    for i in range(0, k):
        if vector[i] < length - k:
            print("error: incorrect top-k element", vector[i])
    vector._del_old()


# CHECK-LABEL: test_sort_stress
fn test_sort_stress():
    print("== test_sort_stress")
    let lens = VariadicList[Int](3, 100, 117, 223, 500, 1000, 1500, 2000, 3000)
    let random_seed = 0
    seed(random_seed)

    @parameter
    fn test[
        cmp_fn: fn[type: AnyRegType] (type, type) capturing -> Bool,
        check_fn: fn[type: AnyRegType] (type, type) capturing -> Bool,
    ](length: Int):
        var vector = DynamicVector[Int](length)
        for i in range(length):
            vector.push_back(random_si64(-length, length).to_int())

        _quicksort[Int, cmp_fn](vector.data, len(vector))

        # CHECK-NOT: error
        for i in range(length - 1):
            if not check_fn[Int](vector[i], vector[i + 1]):
                print("error: unsorted, seed is", random_seed)
                return

        vector._del_old()

    @parameter
    @always_inline
    fn _gt[type: AnyRegType](lhs: type, rhs: type) -> Bool:
        return rebind[Int](lhs) > rebind[Int](rhs)

    @parameter
    @always_inline
    fn _geq[type: AnyRegType](lhs: type, rhs: type) -> Bool:
        return rebind[Int](lhs) >= rebind[Int](rhs)

    @parameter
    @always_inline
    fn _lt[type: AnyRegType](lhs: type, rhs: type) -> Bool:
        return rebind[Int](lhs) < rebind[Int](rhs)

    @parameter
    @always_inline
    fn _leq[type: AnyRegType](lhs: type, rhs: type) -> Bool:
        return rebind[Int](lhs) <= rebind[Int](rhs)

    for i in range(len(lens)):
        let length = lens[i]
        test[_gt, _geq](length)
        test[_geq, _geq](length)
        test[_lt, _leq](length)
        test[_leq, _leq](length)


fn main():
    test_sort_small_3()
    test_sort_small_5()
    test_sort0()
    test_sort2()
    test_sort3()
    test_sort3_dupe_elements()
    test_sort4()
    test_sort5()
    test_sort_reverse()
    test_sort_semi_random()
    test_sort9()
    test_sort103()
    test_sort_any_103()
    test_quick_sort_repeated_val()

    test_sort_stress()

    # CHECK-LABEL: test_partition_top_k_7_5
    # CHECK-NOT: incorrect top-k
    test_partition_top_k(7, 5)
    # CHECK-LABEL: test_partition_top_k_11_2
    # CHECK-NOT: incorrect top-k
    test_partition_top_k(11, 2)
    # CHECK-LABEL: test_partition_top_k_4_1
    # CHECK-NOT: incorrect top-k
    test_partition_top_k(4, 1)
