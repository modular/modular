from std.memory import stack_allocation


struct WithSimdField:
    var a: UInt8
    var b: UInt8
    var c: UInt16
    var d: UInt32
    var e: SIMD[DType.uint8, 16]  # BUG: should sit at C-compatible offset 8
    var f: UInt32

    def __init__(out self, a: UInt8, b: UInt8, c: UInt16, d: UInt32, f: UInt32):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = SIMD[DType.uint8, 16](0)
        self.f = f


struct WithInlineArrayField:
    var a: UInt8
    var b: UInt8
    var c: UInt16
    var d: UInt32
    var e: InlineArray[UInt8, 16]  # control: matches C layout exactly
    var f: UInt32

    def __init__(out self, a: UInt8, b: UInt8, c: UInt16, d: UInt32, f: UInt32):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = InlineArray[UInt8, 16](fill=0)
        self.f = f


def main() raises:
    var p1 = stack_allocation[2, WithSimdField]()
    var size1 = Int(p1 + 1) - Int(p1)
    var s1 = WithSimdField(1, 2, 3, 4, 5)
    var sp1 = UnsafePointer(to=s1)
    var off1 = Int(UnsafePointer(to=s1.e)) - Int(sp1)
    print("SIMD field: size", size1, "(C-correct: 28), offset e", off1, "(C-correct: 8)")

    var p2 = stack_allocation[2, WithInlineArrayField]()
    var size2 = Int(p2 + 1) - Int(p2)
    var s2 = WithInlineArrayField(1, 2, 3, 4, 5)
    var sp2 = UnsafePointer(to=s2)
    var off2 = Int(UnsafePointer(to=s2.e)) - Int(sp2)
    print("InlineArray field (control): size", size2, ", offset e", off2)

    if size1 != 28 or off1 != 8:
        print("BUG CONFIRMED: SIMD[DType.uint8, 16] struct field does not match C layout")
    else:
        print("no mismatch observed on this build")
