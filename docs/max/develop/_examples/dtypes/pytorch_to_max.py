# DOC: max/develop/dtypes.mdx

import torch
from max.dtype import DType

# PyTorch tensor
pt_tensor = torch.randn(10, 10, dtype=torch.float16)

# Convert PyTorch dtype to MAX dtype
# API: DType.from_torch(dtype)
#   dtype: PyTorch dtype
#   Returns: Corresponding MAX DType
#   Raises: ValueError if dtype not supported
#   Raises: RuntimeError if torch not installed
max_dtype = DType.from_torch(pt_tensor.dtype)
print(f"PyTorch {pt_tensor.dtype} → MAX {max_dtype}")  # float16 → DType.float16
