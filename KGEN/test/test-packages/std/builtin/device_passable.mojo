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
    ](mut self, value: T, target: MutOpaquePointer[_]):
        ...


trait DevicePassable:
    comptime device_type: AnyType

    @staticmethod
    def _is_convertible_to_device_type[T: AnyType]() -> Bool:
        ...

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        ...

    @staticmethod
    def get_type_name() -> String:
        ...
