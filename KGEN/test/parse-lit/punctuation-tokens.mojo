# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo %s -verify-diagnostics

# CHECK: module {
# CHECK-NEXT: }

%     # expected-error {{unexpected token in expression}}
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
