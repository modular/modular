# DOC: max/develop/dtypes.mdx

from max.dtype import DType

# DType is an enum that defines how numbers are stored in tensors
# Access dtypes as attributes of the DType class
print(DType.float32)  # 32-bit floating point
print(DType.int32)  # 32-bit integer
print(DType.bool)  # Boolean values
