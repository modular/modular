# Companion module for m1-3-owned-param-main-vs-import.mojo /
# m1-3-owned-param-via-import.mojo — byte-for-byte the same struct,
# defined here so it can be IMPORTED rather than run directly. See the
# other two files for what this demonstrates.

struct IoEventLike(ImplicitlyCopyable):
    var token: UInt64
    var fd: Int32
    var events: UInt32

    def __init__(out self):
        self.token = 0
        self.fd = -1
        self.events = 0

    def __moveinit__(out self, owned existing: Self):
        self.token = existing.token
        self.fd = existing.fd
        self.events = existing.events
