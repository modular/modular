# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


trait DevicePassable:
    comptime device_type: AnyType

    @staticmethod
    fn _is_convertible_to_device_type[T: AnyType]() -> Bool:
        ...

    fn _to_device_type(self, target: MutOpaquePointer[_]):
        ...

    @staticmethod
    fn get_type_name() -> String:
        ...
