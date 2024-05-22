# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Provides callback data types and utilities."""

from ._kserve_impl import ModelInferRequest, ModelInferResponse
from ._serve_rt import Batch


trait ServerCallbacks(Movable):
    """A trait for containers of server lifecycle callbacks."""

    fn on_server_start(inout self):
        """Called when the server starts."""
        ...

    fn on_server_stop(inout self):
        """Called when the server stops."""
        ...

    fn on_batch_receive(inout self, batch: Batch):
        """Called on receipt of a new batch."""
        ...

    fn on_batch_complete(inout self, start_ns: Int, batch: Batch):
        """Called on completion of a new batch."""
        ...

    fn on_request_receive(inout self, request: ModelInferRequest):
        """Called on receipt of a new request."""
        ...

    fn on_request_ok(inout self, start_ns: Int, request: ModelInferRequest):
        """Called on a successful completion of a new request."""
        ...

    fn on_request_fail(inout self, request: ModelInferRequest):
        """Called on a failed completion of a new request."""
        ...


@value
struct NoopServerCallbacks(ServerCallbacks):
    """A no-op server callback implementation. All implementations are a no-op.
    """

    fn on_server_start(inout self):
        pass

    fn on_server_stop(inout self):
        pass

    fn on_batch_receive(inout self, batch: Batch):
        pass

    fn on_batch_complete(inout self, start_ns: Int, batch: Batch):
        pass

    fn on_request_receive(inout self, request: ModelInferRequest):
        pass

    fn on_request_ok(inout self, start_ns: Int, request: ModelInferRequest):
        pass

    fn on_request_fail(inout self, request: ModelInferRequest):
        pass


struct Guarded[T: ServerCallbacks, enabled: Bool](ServerCallbacks):
    """A utility to control callbacks from a parameter.
    The `enabled` parameter determines whether the callback is called or not.
    """

    var cb: T
    """The underlying ServerCallback instance."""

    fn __init__(inout self, owned cb: T):
        self.cb = cb^

    fn __moveinit__(inout self, owned existing: Self):
        self.cb = existing.cb^

    fn on_server_start(inout self):
        @parameter
        if enabled:
            self.cb.on_server_start()

    fn on_server_stop(inout self):
        @parameter
        if enabled:
            self.cb.on_server_stop()

    fn on_batch_receive(inout self, batch: Batch):
        @parameter
        if enabled:
            self.cb.on_batch_receive(batch)

    fn on_batch_complete(inout self, start_ns: Int, batch: Batch):
        @parameter
        if enabled:
            self.cb.on_batch_complete(start_ns, batch)

    fn on_request_receive(inout self, request: ModelInferRequest):
        @parameter
        if enabled:
            self.cb.on_request_receive(request)

    fn on_request_ok(inout self, start_ns: Int, request: ModelInferRequest):
        @parameter
        if enabled:
            self.cb.on_request_ok(start_ns, request)

    fn on_request_fail(inout self, request: ModelInferRequest):
        @parameter
        if enabled:
            self.cb.on_request_fail(request)


struct CallbacksPair[A: ServerCallbacks, B: ServerCallbacks](ServerCallbacks):
    """A utility struct to chain callbacks together.
    Chaining CallbacksPair enables using multiple callbacks for server events.
    """

    var a: A
    """The first ServerCallback instance."""
    var b: B
    """The second ServerCallback instance."""

    fn __init__(inout self, owned a: A, owned b: B):
        """Initializes the pair with two callbacks."""
        self.a = a^
        self.b = b^

    fn __moveinit__(inout self, owned existing: Self):
        """Move initializes the pair from another CallbacksPair."""
        self.a = existing.a^
        self.b = existing.b^

    @always_inline
    fn on_server_start(inout self):
        self.a.on_server_start()
        self.b.on_server_start()

    @always_inline
    fn on_server_stop(inout self):
        self.a.on_server_stop()
        self.b.on_server_stop()

    @always_inline
    fn on_batch_receive(inout self, batch: Batch):
        self.a.on_batch_receive(batch)
        self.b.on_batch_receive(batch)

    @always_inline
    fn on_batch_complete(inout self, start_ns: Int, batch: Batch):
        self.a.on_batch_complete(start_ns, batch)
        self.b.on_batch_complete(start_ns, batch)

    @always_inline
    fn on_request_receive(inout self, request: ModelInferRequest):
        self.a.on_request_receive(request)
        self.b.on_request_receive(request)

    @always_inline
    fn on_request_ok(inout self, start_ns: Int, request: ModelInferRequest):
        self.a.on_request_ok(start_ns, request)
        self.b.on_request_ok(start_ns, request)

    @always_inline
    fn on_request_fail(inout self, request: ModelInferRequest):
        self.a.on_request_fail(request)
        self.b.on_request_fail(request)


@parameter
fn make_callbacks_pair[
    A: ServerCallbacks, B: ServerCallbacks
](owned a: A, owned b: B) -> CallbacksPair[A, B]:
    """A utility function to create a callback pair from two ServerCallback
    implementations.
    """
    return CallbacksPair[A, B](a^, b^)


@parameter
fn make_callbacks_triple[
    A: ServerCallbacks, B: ServerCallbacks, C: ServerCallbacks
](owned a: A, owned b: B, owned c: C) -> CallbacksPair[CallbacksPair[A, B], C]:
    """A utility function to create a callback triple from three ServerCallback
    implementations.
    """
    return make_callbacks_pair(make_callbacks_pair(a^, b^), c^)


@parameter
fn make_callbacks_quadruple[
    A: ServerCallbacks,
    B: ServerCallbacks,
    C: ServerCallbacks,
    D: ServerCallbacks,
](owned a: A, owned b: B, owned c: C, owned d: D) -> CallbacksPair[
    CallbacksPair[CallbacksPair[A, B], C], D
]:
    """A utility function to create a callback quadruple from four
    ServerCallback implementations.
    """
    return make_callbacks_pair(make_callbacks_triple(a^, b^, c^), d^)


@parameter
fn make_callbacks_quintuple[
    A: ServerCallbacks,
    B: ServerCallbacks,
    C: ServerCallbacks,
    D: ServerCallbacks,
    E: ServerCallbacks,
](owned a: A, owned b: B, owned c: C, owned d: D, owned e: E) -> CallbacksPair[
    CallbacksPair[CallbacksPair[CallbacksPair[A, B], C], D], E
]:
    """A utility function to create a callback quintuple from five
    ServerCallback implementations.
    """
    return make_callbacks_pair(make_callbacks_quadruple(a^, b^, c^, d^), e^)
