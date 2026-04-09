# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated --verify-diagnostics %s


@fieldwise_init
struct Foo:
    def __getitem_param__[T: AnyType](self) raises:
        pass


# expected-note @below {{or mark surrounding function as 'raises'}}
def main():
    var f = Foo()

    # expected-error @below {{cannot call function that may raise in a context that cannot raise}}
    # expected-note @below {{try surrounding the call in a 'try' block}}
    _ = f[Int]
