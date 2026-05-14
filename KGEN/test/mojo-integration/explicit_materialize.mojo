# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo -debug-level full %s | FileCheck %s


def make_int_list() -> List[Int]:
    var l = [1, 2, 3]
    return l^


def make_flat() -> List[StaticString]:
    var l = List[StaticString]()
    l.append("source")
    l.append("target")
    return l^


def make_nested() -> List[List[StaticString]]:
    var outer = List[List[StaticString]]()
    var inner1 = List[StaticString]()
    inner1.append("alpha")
    inner1.append("beta")
    var inner2 = List[StaticString]()
    inner2.append("gamma")
    inner2.append("epsilon")
    outer.append(inner1^)
    outer.append(inner2^)
    return outer^


def main():
    comptime lst = make_int_list()
    var dyn_lst = materialize[lst]()
    # CHECK: [1, 2, 3]
    print(dyn_lst)

    comptime flat = make_flat()
    var names = materialize[flat]()
    # CHECK: [source, target]
    print(names)

    comptime nested = make_nested()
    var nested_names = materialize[nested]()
    # CHECK-LITERAL: [[alpha, beta], [gamma, epsilon]]
    print(nested_names)
