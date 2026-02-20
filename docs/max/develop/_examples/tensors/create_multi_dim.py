# DOC: max/develop/tensors.mdx

from max.tensor import Tensor

# Create a 2-D tensor (a matrix)
matrix = Tensor.constant([[1, 2, 3], [4, 5, 6]])
print(matrix)

# Create a 3-D tensor (a cube of numbers)
cube = Tensor.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(cube)
