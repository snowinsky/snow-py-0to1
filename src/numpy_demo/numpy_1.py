import numpy as np
import math

print(np.__version__)

## 数组
a = np.array([
    [[1, 2, 3, 4, 5], [6, 7, 8, 9, 0]],
    [[11, 21, 31, 41, 51], [61, 71, 81, 91, 1]],
    [[12, 22, 32, 42, 52], [62, 72, 82, 92, 2]]
], dtype=np.float64)
print(a.view(dtype=np.float64))  # a 是个 view，能打印出整个数组
print(a.base)  # a 不是一个副本，所以打印出 None
print('a.ndim={}'.format(a.ndim))  # a 是 2 维数组，所以打印出 2
print('a.size={}'.format(a.size))  # a这个数组所有的元素的总数
print('a.shape={} 表示这是个{}维数组，每个维度{}个元素'.format(a.shape, a.shape[0], a.shape[1]))
print('a.size--{} = a.shape---{}中各个值的乘积'.format(a.size, math.prod(a.shape)))
print('a.dtype={} 数组只有唯一的一种数据类型'.format(a.dtype))

all_0_array = np.zeros((3, 2, 5))
all_1_array = np.ones((3, 2, 5))
all_radom_array = np.empty((3, 2, 5))  # 1维的随机，大于 1 维基本上就是补 0 了。
print(all_0_array)
print(all_1_array)
print(all_radom_array)

print(a, a.reshape(5, 3, 2))

help(np.add)

A = np.array([[1, 2, 3], [3, 4, 5]])  # 2x2
B = np.array([[6, 7, 8, 9, 0], [7, 8, 9, 0, 1], [9, 0, 1, 2, 3]])  # 2x2
C = np.matmul(A, B)  # 或 C = A @ B
print(C)
# 输出
# [[47 23 29 15 11]
#  [91 53 65 37 19]]
