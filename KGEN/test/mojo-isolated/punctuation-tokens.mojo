# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -verify-diagnostics

# CHECK: module {
# CHECK-NEXT: }

# expected-error @below {{TODO: expressions are not yet supported at the file scope level}}
# expected-error @below {{unexpected token in expression}}
%
&
(
)
*
+
,
-
.
/
:
;
<
=
>
@
[
]
^
{
|
}
~
_
!=
%=
&=
**
*=
+=
-=
->
//
/=
:=
<<
<=
<>
==
>=
>>
@=
^=
|=
**=
...
//=
<<=
>>=
