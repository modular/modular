# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -verify-diagnostics %s


def use(a: Int):
    pass


def use(a: StaticString):
    pass


def test():
    # True dead code - check for warnings
    # expected-warning @+1 {{unreachable code on right side of 'False and ...'}}
    use(0 and 0)
    # expected-warning @+1 {{unreachable code on right side of 'False and ...'}}
    use(0 and 1)
    # expected-warning @+1 {{unreachable code on right side of 'True or ...'}}
    use(1 or 0)
    # expected-warning @+1 {{unreachable code on right side of 'True or ...'}}
    use(1 or 1)

    # expected-warning @+1 {{right hand side expression of 'if True' is dead}}
    use(1 if 1 else 0)
    # expected-warning @+1 {{left hand side expression of 'if False' is dead}}
    use(1 if 0 else 0)

    # expected-warning @+1 {{'if' condition always evaluates to 'False'; 'if' branch is unreachable}}
    if 0:
        use("dead")
    # expected-warning @+1 {{'if' condition always evaluates to 'False'; 'if' branch is unreachable}}
    elif 0:
        use("dead elif")
    # expected-warning @+1 {{'if' condition always evaluates to 'True'; 'else' branch is unreachable}}
    elif 1:
        use("live elif")
    else:
        use("dead else")

    # No dead user code, but still constant branch conditions
    # expected-warning @+1 {{constant value on left side of 'False or ...'}}
    use(0 or 0)
    # expected-warning @+1 {{constant value on left side of 'False or ...'}}
    use(0 or 1)
    # expected-warning @+1 {{constant value on left side of 'True and ...'}}
    use(1 and 0)
    # expected-warning @+1 {{constant value on left side of 'True and ...'}}
    use(1 and 1)
