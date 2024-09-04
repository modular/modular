# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Imports from 'bindings.so'
import bindings

if __name__ == "__main__":
    pass

    # print(bindings)

    result = bindings.mojo_count_args(1, 2)

    print("Result from Mojo 🔥:", result)
