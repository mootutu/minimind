import torch
import torch.nn as nn

# Part1: nn.Dropout() 层

# dropout_layer = nn.Dropout(p=0.5)

# t1 = torch.Tensor([1, 2, 3])
# t2 = dropout_layer(t1)
# 这里 Dropout 丢弃是为了保持期望不变，将其他部分扩大两倍
# print(t2)

# Part2: nn.Linear() 层

# layer = nn.Linear(in_features=3, out_features=5, bias=True)
# t1 = torch.Tensor([1, 2, 3]) # shape: (3, )

# t2 = torch.Tensor([[[1, 2, 3]]]) # shape: (1, 3)
# 这里应用的 w 和 b 是随机的，真实训练里会在 optimizer 上更新
# output2 = layer(t2) # shape: (1, 5)
# print(output2)
# 线性变化，就是对应用的张量乘以一个 w 矩阵再加上一个偏置

# Part3: view() 函数

# t = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]) # shape: (2, 6)
# print(id(t))
# t_view1 = t.view(3, 4)
# print(t_view1) # shape: (3, 4)
# print(id(t_view1))
# t_view2 = t.view(4, 3)
# print(t_view2) # shape: (4, 3)
# print(id(t_view2))


# Part4: transpose() 函数

# t1 = torch.Tensor([[1, 2, 3], [4, 5, 6]]) # shape: (2, 3)
# print(t1)
# t1 = t1.transpose(0, 1) # shape: (3, 2)
# print(t1)


# Part5: triu() 函数
# diagonal 表示主对角线的偏移量，默认值为 0

# x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) # shape: (3, 3)
# print(x)
# print(torch.triu(x))
# Output: tensor([[1, 2, 3],
#                 [0, 5, 6],
#                 [0, 0, 9]])

# print(torch.triu(x, diagonal=1))
# Ouput: tensor([[0, 2, 3],
#                 [0, 0, 6],
#                 [0, 0, 0]])   

# print(torch.triu(x, diagonal=-1))
# Ouput: tensor([[1, 2, 3],
#                 [4, 5, 6],
#                 [0, 8, 9]])   

# Part6: reshape() 函数

# x = torch.arange(1, 7) # tensor([1, 2, 3, 4, 5, 6])

# y = torch.reshape(x, (2, 3))
print(y)
# Output: tensor([[1, 2, 3],
#                 [4, 5, 6]])

# 使用 -1 自动推理
# z = torch.reshape(x, (2, -1))
# print(z)
# Output: tensor([[1, 2, 3],
#                 [4, 5, 6]])

# view 和 reshape 函数的区别？
