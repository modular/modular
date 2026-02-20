# DOC: max/develop/tensors.mdx

from max.tensor import Tensor

t = Tensor.constant([[1, 2, 3], [4, 5, 6]])
print(t.num_elements())  # 6 (2 rows x 3 columns)
