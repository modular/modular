# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s | FileCheck %s

from collections import Dict

# COM: testing MemoryBlob refCount in the interpreter
# COM: where to create the alias of the Dict, we have
# COM: two exact same attributes to represent the two ["001"]s
# COM: as the Dict values, but the interpreter only creates
# COM: one MemoryBlob for this memory type allocation due to
# COM: mlir::Attribute uniquing. Use refCount to make sure we free
# COM: these blobs correctly if needed in the interpreter.
alias COUNTRY_CODE_TO_REGION_CODE: Dict[Int, List[String]] = {
    800: ["001"],
    808: ["001"],
}


fn _get_country_codes() -> List[String]:
    var result = List[String]()
    for value in COUNTRY_CODE_TO_REGION_CODE.values():
        result += value
    return result


def main():
    var COUNTRY_CODES = _get_country_codes()
    for v in COUNTRY_CODES:
        print(v)


# CHECK: 001
# CHECK: 001
