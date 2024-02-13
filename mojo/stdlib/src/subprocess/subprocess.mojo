# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements the subprocess package."""


from sys import external_call
from collections import DynamicVector
from builtin.file import _OwnedStringRef


fn _popen(cmd: String, mode: String = "r") -> Pointer[NoneType]:
    return external_call["popen", Pointer[NoneType]](
        cmd._as_ptr(), mode._as_ptr()
    )


fn _pclose(strm: Pointer[NoneType]) -> Int32:
    return external_call["pclose", Int32](strm)


fn _read(strm: Pointer[NoneType]) raises -> String:
    var len: Int = 0
    var line = DTypePointer[DType.int8]()
    var res = String("")

    while True:
        var read = external_call["getline", Int](
            Pointer.address_of(line), Pointer.address_of(len), strm
        )
        if read == -1:
            break
        res += StringRef(line, read)

    if line:
        external_call["free", NoneType](line)

    return res.rstrip()


fn run(cmd: String) raises -> String:
    let hdl = _popen(cmd)
    if not hdl:
        raise "unable to execute the command `" + cmd + "`"
    let res = _read(hdl)
    _ = _pclose(hdl)
    return res
