# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


trait DevicePassable:
    comptime device_type: AnyType

    fn _to_device_type(self, target: MutOpaquePointer[_]):
        ...

    @staticmethod
    fn get_type_name() -> String:
        ...
