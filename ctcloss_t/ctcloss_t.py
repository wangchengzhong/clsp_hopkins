# -*- coding: utf-8 -*-

import torch

if __name__ == '__main__':
    T = 50  # 输入序列长度
    C = 20  # 分类总数量，包括blank
    N = 16  # Batch Size
    S = 30  # 在当前Batch中最长的目标序列在padding之后的长度
    S_min = 10  # 最小的目标序列长度

    # 初始化一个随机输入序列，形状为(T,N,C)=>(50,16,20)
    input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()

    # 初始化一个随机目标序列，blank=0,1:C=classes，形状为(N,S)=>(16,30)
    target = torch.randint(low=1, high=C, size=(N, S), dtype=torch.long)

    # 初始化输入序列长度Tensor，形状为N，值为T
    input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)

    # 初始化一个随机目标序列长度，形状为N，值最小为10，最大为30
    target_lengths = torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long)

    # 创建一个CTCLoss对象
    ctc_loss = torch.nn.CTCLoss()

    # 调用CTCLoss()对象计算损失值
    loss = ctc_loss(input, target, input_lengths, target_lengths)

    print(loss)
