# DOC: max/develop/tensors.mdx

from max.tensor import Tensor

# 1-D tensor
x = Tensor.constant([1, 2, 3, 4])
print(x.shape)

# 2-D tensor
matrix = Tensor.constant([[1, 2, 3], [4, 5, 6]])
print(matrix.shape)

# 3-D tensor
cube = Tensor.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(cube.shape)
