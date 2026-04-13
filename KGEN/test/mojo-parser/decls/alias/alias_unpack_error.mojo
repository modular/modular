# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -import-mojo -verify-diagnostics -split-input-file %s

# expected-error @below {{cannot unpack value of 'Tuple[Int, FloatDyn]' of 2 elements into 3 values}}
comptime a, (b, c, d) = (1, (2, 3.0))

# // -----

# expected-error @below {{invalid comptime declaration: expected an identifier or '_'}}
comptime t, True, c = 1, 2, 3


# // -----


struct A:
    # expected-error @below {{'comptime' constants inside structs must be declared separately; break this into individual declarations}}
    comptime a, b = 1, 2


# // -----


trait A:
    # expected-error @below {{a trait's associated types must be declared separately; break this into individual declarations}}
    comptime a, b = 1, 2


# // -----


# expected-note @below {{previous definition here}}
comptime a, b = 1, 2
# expected-error @below {{invalid redefinition of 'b'}}
comptime b, c = 2, 3
