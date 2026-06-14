#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kong Xiaoshuang
Date: 6/13/26
Description: Losses
"""
import torch
import torch.nn.functional as F


def info_nce_loss(h, h_prime, temperature: float = 1.0):
    """h,h_prime:[B,H], 第 i 行互为正对。 返回标量loss """
    # L2 归一化后，点积即 cos 相似度
    h = F.normalize(h, dim=-1)
    h_prime = F.normalize(h_prime, dim=-1)

    # 相似度矩阵 [B, B]: sim[i,j] = cos(h_i, h_prime_j)
    sim = h @ h_prime.t() / temperature

    # 对角线是正对,其余是负对 -> 等价于"每行的正确类别是 i"的交叉熵
    B = h.size(0)
    labels = torch.arange(B, device=h.device)
    loss_i = F.cross_entropy(sim, labels)  # h -> h_prime 方向
    loss_j = F.cross_entropy(sim.t(), labels)  # h_prime -> h 方向
    return (loss_i + loss_j) / 2


def test():
    torch.manual_seed(0)
    B, H = 8, 768

    # 验证关键性质: 正对越像, loss 越小
    print("验证 InfoNCE 正确性（正对越接近,loss应越小）")

    # 情况1：正对完全随机 不相关
    h = torch.randn(B, H)
    h_rand = torch.randn(B, H)
    print(f"正对随机无关:loss = {info_nce_loss(h, h_rand).item():.4f}")

    # 情况2：正对略相似，加噪声的同向量
    h_near = h + 0.5 * torch.randn(B, H)
    print(f"正对略相似: loss = {info_nce_loss(h, h_near).item():.4f}")

    # 情况3：正对几乎相同
    h_same = h + 0.01 * torch.randn(B, H)
    print(f"正对几乎相同: loss = {info_nce_loss(h, h_same).item():.4f}")
    print("若 loss 依次递减 -> InfoNCE 写对了(拉近正对确实降低 loss)")


def main():
    test()


if __name__ == "__main__":
    main()
