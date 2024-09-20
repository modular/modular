# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Provides callback utilities."""

from utils import unroll
from memory import UnsafePointer


trait ServerCallbacks(Movable):
    """A trait for containers of server lifecycle callbacks."""

    fn on_server_start(inout self):
        """Called when the server starts."""
        ...

    fn on_server_stop(inout self):
        """Called when the server stops."""
        ...

    fn on_batch_receive(inout self, batch_size: Int):
        """Called on receipt of a new batch."""
        ...

    fn on_batch_complete(inout self, start_ns: Int, batch_size: Int):
        """Called on completion of a new batch."""
        ...

    fn on_request_receive(inout self):
        """Called on receipt of a new request."""
        ...

    fn on_request_ok(inout self, start_ns: Int):
        """Called on a successful completion of a new request."""
        ...

    fn on_request_fail(inout self, error: String):
        """Called on a failed completion of a new request."""
        ...


@value
struct NoopServerCallbacks(ServerCallbacks):
    """A no-op server callback implementation. All implementations are a no-op.

    This is the default callback used when no callback operation is required.
    """

    fn on_server_start(inout self):
        pass

    fn on_server_stop(inout self):
        pass

    fn on_batch_receive(inout self, batch_size: Int):
        pass

    fn on_batch_complete(inout self, start_ns: Int, batch_size: Int):
        pass

    fn on_request_receive(inout self):
        pass

    fn on_request_ok(inout self, start_ns: Int):
        pass

    fn on_request_fail(inout self, error: String):
        pass


struct Guarded[T: ServerCallbacks, enabled: Bool](ServerCallbacks):
    """A utility to control callbacks from a parameter.

    This struct wraps an existing callback with a guard that is controlled
    through a parameter `enabled`.

    Parameters:
        T: The server callback implementation to be wrapped.
        enabled:  Determines whether the callback is called or not.
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

    fn on_batch_receive(inout self, batch_size: Int):
        @parameter
        if enabled:
            self.cb.on_batch_receive(batch_size)

    fn on_batch_complete(inout self, start_ns: Int, batch_size: Int):
        @parameter
        if enabled:
            self.cb.on_batch_complete(start_ns, batch_size)

    fn on_request_receive(inout self):
        @parameter
        if enabled:
            self.cb.on_request_receive()

    fn on_request_ok(inout self, start_ns: Int):
        @parameter
        if enabled:
            self.cb.on_request_ok(start_ns)

    fn on_request_fail(inout self, error: String):
        @parameter
        if enabled:
            self.cb.on_request_fail(error)


struct CallbackSet[*Ts: ServerCallbacks](ServerCallbacks):
    """A utility struct to chain callbacks together."""

    # N.B. Using a variadic parmaeter type is not possible as a variable within
    # a struct, so some of the implementation of Tuple is cloned here. It does
    # not support getting individual items or anything else. That you might
    # find in a tuple, however.  In the future, this should be converted to use
    # a standardized type.

    alias _mlir_type = __mlir_type[
        `!kgen.pack<:!kgen.variadic<`,
        ServerCallbacks,
        `> `,
        +Ts,
        `>`,
    ]
    var storage: Self._mlir_type

    @always_inline("nodebug")
    fn __init__(inout self, owned *args: *Ts):
        self = Self(storage=args^)

    @always_inline("nodebug")
    fn __init__(
        inout self,
        *,
        owned storage: VariadicPack[_, ServerCallbacks, Ts],
    ):
        # Mark 'self.storage' as being initialized so we can work on it.
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(self.storage)
        )

        @parameter
        fn initialize_elt[idx: Int]():
            UnsafePointer.address_of(storage[idx]).move_pointee_into(
                UnsafePointer.address_of(self[idx])
            )

        unroll[initialize_elt, Self.__len__()]()
        storage._is_owned = False

    fn __del__(owned self):
        @parameter
        fn destroy_elt[idx: Int]():
            UnsafePointer.address_of(self[idx]).destroy_pointee()

        unroll[destroy_elt, Self.__len__()]()

    @always_inline("nodebug")
    fn __moveinit__(inout self, owned existing: Self):
        # Mark 'storage' as being initialized so we can work on it.
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(self.storage)
        )

        @parameter
        fn initialize_elt[idx: Int]():
            UnsafePointer.address_of(existing[idx]).move_pointee_into(
                UnsafePointer.address_of(self[idx])
            )

        unroll[initialize_elt, Self.__len__()]()

    @always_inline
    @staticmethod
    fn __len__() -> Int:
        @parameter
        fn variadic_size(
            x: __mlir_type[`!kgen.variadic<`, ServerCallbacks, `>`]
        ) -> Int:
            return __mlir_op.`pop.variadic.size`(x)

        alias result = variadic_size(Ts)
        return result

    @always_inline("nodebug")
    fn __getitem__[
        idx: Int
    ](ref [_]self: Self) -> ref [__lifetime_of(self)] Ts[idx.value]:
        var storage_kgen_ptr = UnsafePointer.address_of(self.storage).address
        var elt_kgen_ptr = __mlir_op.`kgen.pack.gep`[index = idx.value](
            storage_kgen_ptr
        )
        return UnsafePointer(elt_kgen_ptr)[]

    @always_inline
    fn on_server_start(inout self):
        @parameter
        fn apply[i: Int]():
            self[i].on_server_start()

        unroll[apply, Self.__len__()]()

    @always_inline
    fn on_server_stop(inout self):
        alias length = len(VariadicList(Ts))

        @parameter
        fn apply[i: Int]():
            self[i].on_server_stop()

        unroll[apply, Self.__len__()]()

    @always_inline
    fn on_batch_receive(inout self, batch_size: Int):
        alias length = len(VariadicList(Ts))

        @parameter
        fn apply[i: Int]():
            self[i].on_batch_receive(batch_size)

        unroll[apply, Self.__len__()]()

    @always_inline
    fn on_batch_complete(inout self, start_ns: Int, batch_size: Int):
        alias length = len(VariadicList(Ts))

        @parameter
        fn apply[i: Int]():
            self[i].on_batch_complete(start_ns, batch_size)

        unroll[apply, Self.__len__()]()

    @always_inline
    fn on_request_receive(inout self):
        alias length = len(VariadicList(Ts))

        @parameter
        fn apply[i: Int]():
            self[i].on_request_receive()

        unroll[apply, Self.__len__()]()

    @always_inline
    fn on_request_ok(inout self, start_ns: Int):
        alias length = len(VariadicList(Ts))

        @parameter
        fn apply[i: Int]():
            self[i].on_request_ok(start_ns)

        unroll[apply, Self.__len__()]()

    @always_inline
    fn on_request_fail(inout self, error: String):
        alias length = len(VariadicList(Ts))

        @parameter
        fn apply[i: Int]():
            self[i].on_request_fail(error)

        unroll[apply, Self.__len__()]()
