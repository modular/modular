# Companion module for via_import.mojo: byte-for-byte the same struct as
# the one defined inline in main_vs_import.mojo, placed here so it can be
# IMPORTED rather than compiled as the entry file. The legacy
# `__moveinit__` spelling below was removed in 1.0.0b1 (unified-__init__
# migration), yet this file still compiles fine when imported. See
# main_vs_import.mojo for the full write-up.

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
