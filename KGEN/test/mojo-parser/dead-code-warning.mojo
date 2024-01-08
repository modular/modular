# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo -verify-diagnostics %s


fn main():
    # True dead code - check for warnings
    # expected-warning @+1 {{unreachable code on right side of 'False and ...'}}
    print(0 and 0)
    # expected-warning @+1 {{unreachable code on right side of 'False and ...'}}
    print(0 and 1)
    # expected-warning @+1 {{unreachable code on right side of 'True or ...'}}
    print(1 or 0)
    # expected-warning @+1 {{unreachable code on right side of 'True or ...'}}
    print(1 or 1)

    # expected-warning @+1 {{right hand side expression of 'if True' is dead}}
    print(1 if 1 else 0)
    # expected-warning @+1 {{left hand side expression of 'if False' is dead}}
    print(1 if 0 else 0)

    # expected-warning @+1 {{if statement with constant condition 'if False'}}
    if 0:
        print("dead")
    # expected-warning @+1 {{if statement with constant condition 'if False'}}
    elif 0:
        print("dead elif")
    # expected-warning @+1 {{if statement with constant condition 'if True'}}
    elif 1:
        print("live elif")
    else:
        print("dead else")

    # No dead user code, but still constant branch conditions
    # expected-warning @+1 {{constant value on left side of 'False or ...'}}
    print(0 or 0)
    # expected-warning @+1 {{constant value on left side of 'False or ...'}}
    print(0 or 1)
    # expected-warning @+1 {{constant value on left side of 'True and ...'}}
    print(1 and 0)
    # expected-warning @+1 {{constant value on left side of 'True and ...'}}
    print(1 and 1)
