from std.sys.intrinsics import inlined_assembly
from std.memory import UnsafePointer
from externs import shim_a, shim_b

comptime BYTE_PTR = UnsafePointer[Byte, MutAnyOrigin]


@export("my_entry")
def my_entry(ud: BYTE_PTR) abi("C"):
    pass


@export("my_hook")
def my_hook(ud: BYTE_PTR) abi("C"):
    pass


def entry_pointer[name: String]() -> BYTE_PTR:
    comptime asm_str = (
        "adrp ${0:x}, _" + name + "@PAGE\n"
        "add ${0:x}, ${0:x}, _" + name + "@PAGEOFF\n"
    )
    var addr = inlined_assembly[asm_str, UInt, constraints="=r"]()
    return BYTE_PTR(unsafe_from_address=Int(addr))


def do_work(runtime_int: Int):
    # The crash: in a RAISING function, two extern calls, each passing a
    # runtime Int variable AND an entry_pointer (inline asm) value.
    var rc = shim_a(
        entry_pointer["my_entry"](), runtime_int, Int(8), entry_pointer["my_hook"]()
    )
    shim_b(
        entry_pointer["my_entry"](), runtime_int, Int(10), entry_pointer["my_hook"]()
    )


def main() raises:
    do_work(Int(0x100000000))
    print("ok")
