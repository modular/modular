# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Provides callback data types and utilities."""


trait ServerCallbacks(Movable):
    """A trait for containers of server lifecycle callbacks."""

    fn on_batch_receive(inout self):
        """Called on receipt of a new batch."""
        ...

    fn on_batch_complete(inout self, start_ns: Int):
        """Called on completion of a new batch."""
        ...

    fn on_request_receive(inout self):
        """Called on receipt of a new batch."""
        ...

    fn on_request_ok(inout self, start_ns: Int):
        """Called on a successful completion of a new request."""
        ...

    fn on_request_fail(inout self):
        """Called on a failed completion of a new request."""
        ...


@value
struct NoopServerCallbacks(ServerCallbacks):
    fn on_batch_receive(inout self):
        pass

    fn on_batch_complete(inout self, start_ns: Int):
        pass

    fn on_request_receive(inout self):
        pass

    fn on_request_ok(inout self, start_ns: Int):
        pass

    fn on_request_fail(inout self):
        pass


struct _SC2[A: ServerCallbacks, B: ServerCallbacks](ServerCallbacks):
    var a: A
    var b: B

    fn __init__(inout self, owned a: A, owned b: B):
        self.a = a^
        self.b = b^

    fn __moveinit__(inout self, owned existing: Self):
        self.a = existing.a^
        self.b = existing.b^

    @always_inline
    fn on_batch_receive(inout self):
        self.a.on_batch_receive()
        self.b.on_batch_receive()

    @always_inline
    fn on_batch_complete(inout self, start_ns: Int):
        self.a.on_batch_complete(start_ns)
        self.b.on_batch_complete(start_ns)

    @always_inline
    fn on_request_receive(inout self):
        self.a.on_request_receive()
        self.b.on_request_receive()

    @always_inline
    fn on_request_ok(inout self, start_ns: Int):
        self.a.on_request_ok(start_ns)
        self.b.on_request_ok(start_ns)

    @always_inline
    fn on_request_fail(inout self):
        self.a.on_request_fail()
        self.b.on_request_fail()


@parameter
fn make_callbacks_pair[
    A: ServerCallbacks, B: ServerCallbacks
](owned a: A, owned b: B) -> _SC2[A, B]:
    return _SC2[A, B](a^, b^)


@parameter
fn make_callbacks_triple[
    A: ServerCallbacks, B: ServerCallbacks, C: ServerCallbacks
](owned a: A, owned b: B, owned c: C) -> _SC2[_SC2[A, B], C]:
    return make_callbacks_pair(make_callbacks_pair(a^, b^), c^)


@parameter
fn make_callbacks_quadruple[
    A: ServerCallbacks,
    B: ServerCallbacks,
    C: ServerCallbacks,
    D: ServerCallbacks,
](owned a: A, owned b: B, owned c: C, owned d: D) -> _SC2[
    _SC2[_SC2[A, B], C], D
]:
    return make_callbacks_pair(make_callbacks_triple(a^, b^, c^), d^)


@parameter
fn make_callbacks_quintuple[
    A: ServerCallbacks,
    B: ServerCallbacks,
    C: ServerCallbacks,
    D: ServerCallbacks,
    E: ServerCallbacks,
](owned a: A, owned b: B, owned c: C, owned d: D, owned e: E) -> _SC2[
    _SC2[_SC2[_SC2[A, B], C], D], E
]:
    return make_callbacks_pair(make_callbacks_quadruple(a^, b^, c^, d^), e^)
