comptime CollectionElement = Copyable & ImplicitlyDestructible

struct Matrix[T: CollectionElement]:
    var width: Int
    var height: Int
    var depth: Int

    var _store: List[Self.T]

    fn idx(self, x: Int, y: Int, z: Int) -> Int:
        return z * self.width * self.height + y * self.width + x

    fn __init__(out self, width: Int, height: Int, depth: Int, *, fill: Self.T):
        self.width, self.height, self.depth = width, height, depth
        self._store = List[Self.T](length=width * height, fill=fill)

    fn __getitem__(self, x: Int, y: Int, z: Int) -> Self.T:
        return self._store[self.idx(x, y, z)].copy()

    fn __setitem__(mut self, x: Int, y: Int, z: Int, var value: Self.T):
        self._store[self.idx(x, y, z)] = value^


fn main():
    var m = Matrix[Int](3, 2, 2, fill=0)  # 3 wide, 4 tall, 5 deep
    m[2, 1, 1] = 42
    print(m[2, 1, 1])  # prints 42
