# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -verify-diagnostics

trait Trait1:
    fn f1(self):
        ...

trait Trait2:
    fn f2(self):
        ...

trait Trait3:
    fn f3(self):
        ...

alias Traits12 = Trait1 & Trait2
alias Traits23 = Trait2 & Trait3
alias Traits123 = Trait1 & Trait2 & Trait3

@fieldwise_init
struct Struct4():
    fn f4(self):
        pass

# expected-note @below {{function declared here}}
fn use1[T: Trait1](x: T):
    pass

# Use aliased trait composition.
# expected-note @below {{function declared here}}
fn use12Alias[T: Traits12](x: T):
    pass

# Use direct trait composition.
# expected-note @below {{function declared here}}
fn use12Direct[T: Trait1 & Trait2](x: T):
    pass

# CHECK: lit.fn @"main_use()"
fn main_use():
    s4 = Struct4()

    # expected-error @below {{invalid call to 'use1': could not deduce parameter 'T' of callee 'use1'}}
    # expected-note @below {{failed to infer parameter 'T', argument type 'Struct4' does not conform to trait 'Trait1'}}
    use1(s4)

    # expected-error @below {{invalid call to 'use12Alias': could not deduce parameter 'T' of callee 'use12Alias'}}
    # expected-note @below {{failed to infer parameter 'T', argument type 'Struct4' does not conform to trait 'Trait1 & Trait2'}}
    use12Alias(s4)

    # expected-error @below {{invalid call to 'use12Direct': could not deduce parameter 'T' of callee 'use12Direct'}}
    # expected-note @below {{failed to infer parameter 'T', argument type 'Struct4' does not conform to trait 'Trait1 & Trait2'}}
    use12Direct(s4)
