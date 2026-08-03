# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


struct DeviceBuffer:
    ...


struct DevicePointer:
    ...


trait DeviceTypeEncoder:
    def encode[
        ValueType: AnyType
    ](mut self, value: ValueType, dst: MutOpaquePointer[_]):
        """Stub bit-copy encode for test-package IR emission."""
        pass

    def encode_bits[
        DeviceType: AnyType,
        ValueType: ImplicitlyCopyable,
    ](self, value: ValueType, target: MutOpaquePointer[_]):
        ...

    def encode_device_buffer(
        self, value: DeviceBuffer, target: MutOpaquePointer[_]
    ):
        ...

    def encode_device_ptr(
        self, value: DevicePointer, target: MutOpaquePointer[_]
    ):
        ...

    def encode_fields[
        T: AnyType,
        //,
        DeviceStructType: AnyType = T,
    ](mut self, value: T, target: MutOpaquePointer[_]):
        ...

    def encode_closure_state[
        T: AnyType,
        //,
        DeviceStructType: AnyType,
    ](mut self, value: T, target: MutOpaquePointer[_]):
        ...


trait DevicePassable:
    comptime device_type: AnyType

    @staticmethod
    def _is_convertible_to_device_type[SrcT: AnyType]() -> Bool:
        comptime if Self != Self.device_type and conforms_to(
            Self.device_type, DevicePassable
        ):
            return Self.device_type._is_convertible_to_device_type[SrcT]()
        else:
            return SrcT == Self.device_type

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        ...

    @staticmethod
    def get_type_name() -> String:
        ...
