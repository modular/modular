# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Negative tests for type refinement — verify refinement does NOT incorrectly
# propagate.
#
# RUN: %parse-mojo-isolated -verify-diagnostics %s


trait ProcessTrait:
    def process(self) -> Int:
        ...


trait LeakTestTrait:
    def leak_method(self) -> Int:
        ...


# --- Refinement should NOT leak to a different parameter ---

# expected-note @below {{function declared here}}
def process_element_1[T: ProcessTrait](elem: T) -> Int:
    return elem.process()


def test_wrong_param[T: ImplicitlyCopyable, U: ImplicitlyCopyable](
    t_val: T, u_val: U,
) -> Int where conforms_to(T, ProcessTrait):
    return process_element_1(u_val)  # expected-error {{cannot be converted}}


# --- No refinement without where clause ---

# expected-note @below {{function declared here}}
def process_element_2[T: ProcessTrait](elem: T) -> Int:
    return elem.process()


def test_no_where_clause[T: ImplicitlyCopyable](val: T) -> Int:
    return process_element_2(val)  # expected-error {{cannot be converted}}


# --- Ambiguous method from declared and refined traits ---

trait TraitWithValue:
    # expected-note @below {{candidate declared here}}
    def get_value(self) -> Int:
        ...


trait TraitWithDoubleValue:
    # expected-note @below {{candidate declared here}}
    def get_value(self) -> Int:
        ...


def test_ambiguous_method[T: TraitWithValue](
    x: T,
) -> Int where conforms_to(T, TraitWithDoubleValue):
    return x.get_value()  # expected-error {{ambiguous call to 'get_value'}}


# --- comptime assert on unrelated condition should NOT refine ---

# expected-note @below {{function declared here}}
def process_element_3[T: ProcessTrait](elem: T) -> Int:
    return elem.process()


def test_unrelated_assert[T: ImplicitlyCopyable](val: T) -> Int:
    comptime assert True, "always true"
    return process_element_3(val)  # expected-error {{cannot be converted}}


# --- Refinement inside comptime if must NOT leak outside the branch ---

# expected-note @below {{function declared here}}
def needs_leak_trait_1[T: LeakTestTrait](x: T) -> Int:
    return x.leak_method()


def test_refinement_does_not_leak_past_branch[T: ImplicitlyCopyable](
    val: T,
) -> Int:
    comptime if conforms_to(T, LeakTestTrait):
        pass
    return needs_leak_trait_1(val)  # expected-error {{cannot be converted}}


# expected-note @below {{function declared here}}
def needs_leak_trait_2[T: LeakTestTrait](x: T) -> Int:
    return x.leak_method()


def test_refinement_does_not_leak_to_else_branch[T: ImplicitlyCopyable](
    val: T,
) -> Int:
    comptime if conforms_to(T, LeakTestTrait):
        return needs_leak_trait_2(val)
    else:
        return needs_leak_trait_2(val)  # expected-error {{cannot be converted}}


# --- Variable refinement must NOT persist past comptime if ---

# expected-note @below {{function declared here}}
def needs_leak_trait_3[T: LeakTestTrait](x: T) -> Int:
    return x.leak_method()


def test_var_refinement_does_not_persist_past_comptime_if[
    T: ImplicitlyCopyable
](val: T) -> Int:
    var x = val
    comptime if conforms_to(T, LeakTestTrait):
        _ = needs_leak_trait_3(x)
    return needs_leak_trait_3(x)  # expected-error {{cannot be converted}}


# --- Refinement from nested comptime assert must NOT leak past branch ---

# expected-note @below {{function declared here}}
def needs_leak_trait_4[T: LeakTestTrait](x: T) -> Int:
    return x.leak_method()


def test_nested_assert_refinement_does_not_leak[
    T: ImplicitlyCopyable
](val: T) -> Int:
    comptime if conforms_to(T, LeakTestTrait):
        comptime assert conforms_to(T, LeakTestTrait)
        _ = needs_leak_trait_4(val)
    return needs_leak_trait_4(val)  # expected-error {{cannot be converted}}
