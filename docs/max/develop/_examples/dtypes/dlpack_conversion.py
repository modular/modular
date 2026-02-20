# DOC: max/develop/dtypes.mdx

import numpy as np
from max.tensor import Tensor

# Create a NumPy array
np_array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

# Convert to MAX tensor using DLPack (zero-copy when possible)
tensor = Tensor.from_dlpack(np_array)

print(f"NumPy dtype: {np_array.dtype}")  # float32
print(f"MAX tensor dtype: {tensor.dtype}")  # DType.float32
print(f"MAX tensor shape: {tensor.shape}")  # [2, 2]
