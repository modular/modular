// RUN: kgen-opt %s -verify-parameters -verify-diagnostics -split-input-file -o /dev/null

// expected-error @+1 {{number of argument/parameter names and passing kinds does not match: 4 vs. 3}}
#names_passing_kind_mismatch = #lit.arg_param_list<
  ["a", "b", "c", "d"], [pos, pos_or_kw, kw], [], [], [], []
>

// -----

// expected-error @+1 {{pos passing kind cannot follow pos_or_kw}}
#passing_kind_order1 = #lit.arg_param_list<
  ["a", "b", "c", "d"], [pos_or_kw, pos, kw, implicit], [], [], [], []
>

// -----

// expected-error @+1 {{pos_or_kw passing kind cannot follow implicit}}
#passing_kind_order2 = #lit.arg_param_list<
  ["a", "b", "c", "d"], [pos, kw, implicit, pos_or_kw], [], [], [], []
>

// -----

// expected-error @+1 {{kw passing kind cannot follow implicit}}
#passing_kind_order3 = #lit.arg_param_list<
  ["a", "b", "c", "d"], [pos, pos_or_kw, implicit, kw], [], [], [], []
>

// -----

// expected-error @+1 {{there are more default keyword-only arguments/parameters than keyword-only arguments/parameters: 3 vs. 2}}
#too_many_kw_only_defaults = #lit.arg_param_list<
  ["a", "b", "c", "d"], [pos, pos_or_kw, kw, kw], [], [1 : i8, 2 : i8, 3 : i8], [], []
>

// -----

// expected-error @+1 {{there are more default positional arguments/parameters than positional arguments/parameters: 3 vs. 1}}
#too_many_kw_only_defaults = #lit.arg_param_list<
  ["a", "b", "c", "d"], [pos, kw, kw, kw], [1 : i8, 2 : i8, 3 : i8], [], [], []
>

// -----

// expected-error @+1 {{variadic cannot have a default value}}
#variadic_with_default = #lit.arg_param_list<
  ["a", "b", "c", "d"], [pos, pos_or_kw, kw, kw], [1 : i8], [], [1], []
>

// -----

// expected-error @+1 {{variadic index must be less than the number of elements: 4 vs. 4}}
#variadic_idx_too_big = #lit.arg_param_list<
  ["a", "b", "c", "d"], [pos, pos, kw, kw], [], [], [4], []
>

// -----

// expected-error @+1 {{more than 1 variadic pack not allowed in an argument/parameter list}}
#too_many_packs = #lit.arg_param_list<
  ["a", "b", "c", "d"], [pos, pos_or_kw, pos_or_kw, kw], [], [], [], [1, 2]
>

// -----

// expected-error @+1 {{variadic pack cannot have a default value}}
#pack_with_default = #lit.arg_param_list<
  ["a", "b", "c", "d"], [pos, pos_or_kw, kw, kw], [1 : i8], [], [], [1]
>

// -----

// expected-error @+1 {{variadic pack index must be less than the number of elements: 4 vs. 4}}
#pos_only_pack = #lit.arg_param_list<
  ["a", "b", "c", "d"], [pos, pos, kw, kw], [], [], [], [4]
>

// -----

// expected-error @below {{'bind_type' expected a metatyped type value}}
#bind = #lit.bind_type<:type index, []> : !lit.metatype<@Foo>

// -----

// expected-error @below {{'bind_type' result metatype parameter #0 does not match corresponding input parameter}}
#bind = #lit.bind_type<:metatype<@Foo<?>, <index>> T, [?]> : !lit.metatype<@Foo<1>>

// -----

// expected-error @below {{'bind_type' result metatype should have 1 parameter values, but got 0}}
#bind = #lit.bind_type<:metatype<@Foo, <index>> T, [?]> : !lit.metatype<@Foo<1>>

// -----

// expected-error @below {{'bind_type' cannot change the value of parameter #0}}
#bind = #lit.bind_type<:metatype<@Foo<2>> T, []> : !lit.metatype<@Foo<1>>

// -----

// expected-error @below {{'bind_type' result metatype parameter #0 does not match corresponding input parameter}}
#bind = #lit.bind_type<:metatype<@Foo<?>, <index>> T, [2]> : !lit.metatype<@Foo<3>>

// -----

// expected-error @below {{'bind_type' result metatype signature should have 0 input parameters}}
#bind = #lit.bind_type<:metatype<@Foo<?>, <index>> T, [1]> : !lit.metatype<@Foo<1>, <index>>

// -----

// expected-error @below {{result signature parameter #0 expected to be 'index' but got '!kgen.dtype'}}
#bind = #lit.bind_type<:metatype<@Foo<?>, <index>> T, [?]> : !lit.metatype<@Foo<?>, <dtype>>
