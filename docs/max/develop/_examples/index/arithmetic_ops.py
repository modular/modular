# DOC: max/develop/index.mdx

from max.tensor import Tensor

a = Tensor.constant([1.0, 2.0, 3.0])
b = Tensor.constant([4.0, 5.0, 6.0])

c = a + b  # Addition
d = a * b  # Element-wise multiplication

print(c)
print(d)
