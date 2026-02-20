# DOC: max/develop/tensors.mdx

from max.tensor import Tensor

scalar = Tensor.constant([42])  # Rank 1 (it's a 1-element vector)
vector = Tensor.constant([1, 2, 3])  # Rank 1
matrix = Tensor.constant([[1, 2], [3, 4]])  # Rank 2
cube = Tensor.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # Rank 3

print(vector.rank)  # 1
print(matrix.rank)  # 2
print(cube.rank)  # 3
