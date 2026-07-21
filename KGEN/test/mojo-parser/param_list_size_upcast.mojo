# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values | FileCheck %s


struct InlineArr[length: Int](Movable):
    def __init__[*Ts: Movable](out self: InlineArr[Ts.size]):
        pass


struct ListOf[length: Int]:
    var storage: InlineArr[Self.length]

    # CHECK-LABEL: lit.fn @"__init__[KGENParamList[::AnyType & ::Copyable
    #
    # The `arr` local has type `InlineArr[Ts.size]`. The size looks through the
    # upcast, so it prints on the original Copyable pack with no `upcast` wrapper
    # (rather than `param_list.size<:param_list<!AnyType_Movable> upcast(...)>`).
    #
    # CHECK: %arr = lit.var.decl "arr" {{.*}}#InlineArr <:!Int {{.*}}#kgen.param_list.size<:param_list<!AnyType_Copyable_Movable> {{[^>]*}}Ts.values
    def __init__[*Ts: Copyable](var *elts: *Ts, out self: ListOf[Ts.size]):
        var arr = InlineArr.__init__[*Ts]()
        self.storage = arr^
