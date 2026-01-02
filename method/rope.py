import torch

# Part 1: condition 的基本使用
"""
condition 为 True 时，取 x 的值，为 False 时，取 y 的值
"""

# x = torch.tensor([1,2,3,4,5])
# y = torch.tensor([10, 20, 30, 40, 50, ])

# condition = x > 3

# result = torch.where(condition, x, y)

# print(result) # Output: tensor([10, 20, 30, 4, 5])

# Part 2: torch.arange() 函数的使用
"""
torch.arange() 函数用于创建一个张量，包含从 start 到 end（不包含 end）的元素，步长为 step。
"""

# t = torch.arange(0, 10, 2)
# print(t) # Output: tensor([0, 2, 4, 6, 8])

# t2 = torch.arange(5, 0, -1)
# print(t2) # Output: tensor([5, 4, 3, 2, 1])

# Part 3: torch.outer() 函数的使用
"""
torch.outer() 函数用于计算两个向量的外积（outer product）。

外积的结果是一个矩阵，其中每个元素都是两个向量对应位置元素的乘积。

例如，对于向量 a = [1, 2, 3] 和向量 b = [4, 5, 6]，
它们的外积为：

[1*4, 1*5, 1*6]
[2*4, 2*5, 2*6]
[3*4, 3*5, 3*6]

也就是 ab^T
"""

# v1 = torch.tensor([1, 2, 3])
# v2 = torch.tensor([4, 5, 6])
# result = torch.outer(v1, v2)
# print(result)
# Output: tensor([[ 4,  5,  6],
#                 [ 8, 10, 12],
#                 [12, 15, 18]])

# Part 4: torch.cat() 函数的使用
"""
torch.cat() 函数用于在指定维度上连接多个张量。
"""

# t1 = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[13, 14, 15], [16, 17, 18]]])
# t2 = torch.tensor([[[7, 8, 9], [10, 11, 12]], [[19, 20, 21], [22, 23, 24]]])
# print(t1.shape) # Output: torch.Size([2, 2, 3]) 两个2x3的矩阵
# print(t2.shape) # Output: torch.Size([2, 2, 3])
# result = torch.cat((t1, t2), dim=0)
# print(result)
# print(result.shape) # Output: torch.Size([4, 2, 3])
# Output: tensor([[[ 1,  2,  3],
#                  [ 4,  5,  6]],
# 
#                 [[13, 14, 15],
#                  [16, 17, 18]],
# 
#                 [[ 7,  8,  9],
#                  [10, 11, 12]],
# 
#                 [[19, 20, 21],
#                  [22, 23, 24]]])

# result2 = torch.cat((t1, t2), dim=1)
# print(result2)
# print(result2.shape) # Output: torch.Size([2, 4, 3])
# Output: tensor([[[ 1,  2,  3],
#                  [ 4,  5,  6],
#                  [ 7,  8,  9],
#                  [10, 11, 12]],
# 
#                 [[13, 14, 15],
#                  [16, 17, 18],
#                  [19, 20, 21],
#                  [22, 23, 24]]])

# result3 = torch.cat((t1, t2), dim=-1)
# print(result3)
# print(result3.shape) # Output: torch.Size([2, 2, 6])
# Output: tensor([[[ 1,  2,  3,  7,  8,  9],
#                  [ 4,  5,  6, 10, 11, 12]],
# 
#                 [[13, 14, 15, 19, 20, 21],
#                  [16, 17, 18, 22, 23, 24]]])

# Part 5: torch.unsqueeze() 函数的使用
"""
torch.unsqueeze() 函数用于在指定维度上添加一个大小为 1 的新维度（自适应增加一个维度）。
"""
t1 = torch.tensor([1, 2, 3])
t2 = t1.unsqueeze(0)
print(t2) # Output: tensor([[1., 2., 3.]])
print(t2.shape) # Output: torch.Size([1, 3])