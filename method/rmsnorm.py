import torch

# Part 1: torch.rsqrt() 函数的使用
"""
torch.rsqrt() 函数用于计算张量的平方根的倒数。
"""
# 开方求导数
t = torch.rsqrt(torch.tensor(4.0))
print(t) # Output: tensor(0.5000)

# Part 2: torch.ones() 函数的使用
"""
torch.ones() 函数用于创建一个全 1 的张量。
"""
# 创建一个全 1 的张量
t2 = torch.ones(3,4)
print(t2) 
# Output: tensor([[1., 1., 1., 1.],
#                 [1., 1., 1., 1.],
#                 [1., 1., 1., 1.]])