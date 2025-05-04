// RUN: kgen-opt %s -verify-parameters -verify-diagnostics -split-input-file -o /dev/null

// expected-error @+1 {{pos passing kind cannot follow pos_or_kw}}
#passing_kind_order1 = #lit.pog_list<
  [<"a", pos_or_kw, not_vararg>, <"b", pos, not_vararg>, <"c", kw, not_vararg>, <"d", implicit, not_vararg>],
  [], []
>

// -----

// expected-error @+1 {{pos_or_kw passing kind cannot follow implicit}}
#passing_kind_order2 = #lit.pog_list<
  [<"a", pos, not_vararg>, <"b", kw, not_vararg>, <"c", implicit, not_vararg>, <"d", pos_or_kw, not_vararg>],
  [], []
>

// -----

// expected-error @+1 {{kw passing kind cannot follow implicit}}
#passing_kind_order3 = #lit.pog_list<
  [<"a", pos, not_vararg>, <"b", pos_or_kw, not_vararg>, <"c", implicit, not_vararg>, <"d", kw, not_vararg>],
  [], []
>

// -----

// expected-error @+1 {{there are more default keyword-only arguments/parameters than keyword-only arguments/parameters: 3 vs. 2}}
#too_many_kw_only_defaults = #lit.pog_list<
  [<"a", pos, not_vararg>, <"b", pos_or_kw, not_vararg>, <"c", kw, not_vararg>, <"d", kw, not_vararg>],
  [], [1 : i8, 2 : i8, 3 : i8]
>

// -----

// expected-error @+1 {{there are more default positional arguments/parameters than positional arguments/parameters: 3 vs. 1}}
#too_many_kw_only_defaults = #lit.pog_list<
  [<"a", pos, not_vararg>, <"b", kw, not_vararg>, <"c", kw, not_vararg>, <"d", kw, not_vararg>],
  [1 : i8, 2 : i8, 3 : i8], []
>

// -----

// expected-error @+1 {{default value of variadic must be UnknownAttr}}
#variadic_with_default = #lit.pog_list<
  [<"a", pos, not_vararg>, <"b", pos_or_kw, pos_vararg>, <"c", kw, not_vararg>, <"d", kw, not_vararg>],
  [1 : i8], []
>

// -----

// expected-error @+1 {{'inferred' parameter follows non-inferred parameter}}
#too_many_packs = #lit.pog_list<
  [<"a", pos, not_vararg>, <"b", inferred, not_vararg>],
  [], []
>

// -----

// expected-error @+1 {{default value of variadic pack must be UnknownAttr}}
#pack_with_default = #lit.pog_list<
  [<"a", pos, not_vararg>, <"b", pos_or_kw, pack_vararg>, <"c", kw, not_vararg>, <"d", kw, not_vararg>],
  [1 : i8], [], owned_in_mem
>

// -----

// expected-error @below {{pack not supported in parameter list}}
lit.fn @foo() -> !lit<type_signature<"j": variadic<index> pack_vararg>> {
  kgen.return
}
