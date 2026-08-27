from std.memory import UnsafePointer

comptime BYTE_PTR = UnsafePointer[Byte, MutAnyOrigin]


@extern("probe_a")
def probe_a(ptr: BYTE_PTR, a: Int, b: Int, ptr2: BYTE_PTR) abi("C") -> Int32:
    ...


@extern("probe_b")
def probe_b(ptr: BYTE_PTR, a: Int, b: Int, ptr2: BYTE_PTR) abi("C"):
    ...


def shim_a(ptr: BYTE_PTR, a: Int, b: Int, ptr2: BYTE_PTR) -> Int32:
    return probe_a(ptr, a, b, ptr2)


def shim_b(ptr: BYTE_PTR, a: Int, b: Int, ptr2: BYTE_PTR):
    probe_b(ptr, a, b, ptr2)
