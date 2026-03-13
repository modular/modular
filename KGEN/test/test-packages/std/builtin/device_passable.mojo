# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


trait DevicePassable:
    comptime device_type: AnyType

    @staticmethod
    def _is_convertible_to_device_type[T: AnyType]() -> Bool:
        ...

    def _to_device_type(self, target: MutOpaquePointer[_]):
        ...

    @staticmethod
    def get_type_name() -> String:
        ...
