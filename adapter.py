#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kong Xiaoshuang
Date: 6/13/26
Description: Adapter

Stage 2 - 组件1: 自去偏 adapter G(·)
====================================
对应论文 4.2 节 + 5.1.5。

论文原文: "the self-debiasing adapter G(·) as the one-layer MLP with the
ReLU activation function"。输入输出维度都等于 BERT 句向量维度 (768)。

它的作用: 把"有偏的句向量 z"投影到"无偏子空间的表征 h = G(z)"。
训练时 BERT 冻结,只更新这个 adapter 的参数。

本模块只定义结构 + 自检维度,不涉及训练。
"""
import torch
import torch.nn as nn


class DebiasAdapter(nn.Module):
    """单层MLP + RELU 把[B,H] 我句向量映射到同维度的去偏表征"""

    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

    def forward(self, z):
        return self.net(z)


def main():
    # 自检:维度对不对、参数量多少、是否可训练
    H = 768
    adapter = DebiasAdapter(H)

    fake_z = torch.randn(4, H)  # 假装 4 个句向量

    h = adapter(fake_z)
    print(f"输入维度:{tuple(fake_z.shape)}")
    print(f"输出维度:{tuple(h.shape)}  (应与输入相同)")

    n_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    print(f"可训练参数量:{n_params:,} (768*768 +768 bias = {768 * 768 + 768:,})")

    print("结构:")
    print(adapter)



if __name__ == "__main__":
    main()
