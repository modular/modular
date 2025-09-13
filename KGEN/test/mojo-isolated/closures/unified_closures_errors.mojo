# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated -verify-diagnostics %s


trait MyInterface:
    fn thing(self):
        ...


fn make_closure(x: Int) -> Int:
    fn parametric[T: MyInterface](a: T) unified {}:
        # expected-error @below {{use of unknown declaration 'A'}}
        alias X = A
        pass

    return x

struct Mem(Copyable, ImplicitlyCopyable):
    pass

fn use(a:Mem):
    pass


fn foo(a: Mem):
    # expected-error @below {{cannot capture a by copy or move because it is not register passable and your closure is marked as register passable.}}
    fn closure() unified register_passable {var}:
        use(a)


fn bar(a: Mem):
    # expected-error @below {{a function cannot be register passable unless it is unified}}
    fn closure() register_passable {var}:
        use(a)
