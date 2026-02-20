# DOC: max/develop/dtypes.mdx

from max.dtype import DType

# Check memory size of different dtypes
print(f"float32 size: {DType.float32.size_in_bytes} bytes")  # 4
print(f"float32.is_float(): {DType.float32.is_float()}")  # True
print(f"int32.is_integral(): {DType.int32.is_integral()}")  # True
print(f"float8_e4m3fn.is_float8(): {DType.float8_e4m3fn.is_float8()}")  # True
