# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen %s -elaborate -verify-diagnostics


@export
def origins_dont_exist_at_elaboration_time(a: String, b: String) -> Bool:
    # expected-error @+1 {{origin equality may only be tested in 'where' clauses}}
    return origin_of(b).contains[origin_of(a)]
