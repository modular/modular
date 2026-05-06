# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


def main():
    var d = Dict[String, Int]()
    d["one"] = 1
    d["two"] = 2
    d["three"] = 3
    print(len(d))  # bp1: d has 3 live entries

    var d_empty = Dict[String, Int]()
    print(len(d_empty))  # bp2: empty dict
