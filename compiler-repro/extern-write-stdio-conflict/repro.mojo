from std.memory import stack_allocation

comptime OpaqueBuf = UnsafePointer[NoneType, MutAnyOrigin]


@extern("write")
def mjo_write(fd: Int, buf: OpaqueBuf, count: Int) abi("C") -> Int: ...


def main() raises:
    print("this print() pulls in the stdlib's own write() binding")
    var buf = stack_allocation[4, Byte]()
    buf[0] = 65
    var n = mjo_write(1, buf.bitcast[NoneType](), 1)
    print("wrote", n, "bytes via the custom binding")
