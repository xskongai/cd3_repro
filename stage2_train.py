#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kong Xiaoshuang
Date: 6/13/26
Description: stage2_train
Algorithm 2: adapter + infoNCE 对比训练

Stage 2 - 主体: 对比自去偏训练
==============================
对应论文 Algorithm 2。

严格保证三件事(代码里会打印/断言让你亲眼确认):
  1. BERT 冻结 -> 可训练参数 = 0
  2. 只有 adapter 在更新
  3. 去偏发生 -> 训练后反事实句对的 cos 升高(被拉近)

沙箱连不上 HF,所以本文件用 --fake 开关可在无 BERT 时验证训练循环逻辑;
你在 Mac 上直接 `python3 stage2_train.py` 跑真 BERT。

"""
import sys

import pkg_resources
import torch
from torch.optim import Adam
import torch.nn.functional as F

import config
from adapter import DebiasAdapter
from losses import info_nce_loss


def freeze_bert_report(model):
    """ 冻结 BERT 全部参数，并打印可训练参数量(应为0)确认"""
    for p in model.parameters():
        p.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[冻结检查]BERT 可训练参数 = {trainable:,} (应为0)")


@torch.no_grad()
def avg_pair_cos(adapter, z, z_hat):
    """给定句向量对，经adapter后 算平均 cos(衡量正对被拉得多近)"""
    h = F.normalize(adapter(z), dim=-1)
    hh = F.normalize(adapter(z_hat), dim=-1)
    return F.cosine_similarity(h, hh, dim=-1).mean().item()


def build_pairs_with_bert():
    """把冻结的BERT把反事实句对编码成向量对"""
    from stage1_search import encode, _load_model
    from stage1_augment import make_counterfactual_pair

    _, model = _load_model()
    freeze_bert_report(model)

    raw = [
        "The boy wants a dirt bike as a gift.",
        "He is a very intelligent person.",
        "The man worked as a doctor for years.",
        "His father drove to the office.",
        "That guy is good at math.",
        "The son inherited the family business.",
        "He always solved problems quickly.",
        "The male candidate gave a speech.",
        "John is the leader of the team.",
        "He fixed the engine himself.",
        "The boy played football after school.",
        "The man lifted the heavy box.",
    ]

    xs, xhs = [], []
    for s in raw:
        x, xh, _ = make_counterfactual_pair(s)
        if xh:
            xs.append(x)
            xhs.append(xh)

    z = encode(xs)
    z_hat = encode(xhs)

    return z.detach(), z_hat.detach()


@torch.no_grad()
def collapse_report(adapter, z, z_hat):
    """坍缩监控:同时看正对 cos 和负对 cos。
      健康: 正对 cos 高(性别差异被抹平)、负对 cos 低(不同句子仍可区分)
      坍缩: 两者都 -> 1(所有向量压成一个点,模型啥都不分了)
    返回 (pos_cos, neg_cos)。
    """
    h = torch.nn.functional.normalize(adapter(z), dim=-1)
    hh = torch.nn.functional.normalize(adapter(z_hat), dim=-1)
    # 正对:对角线
    pos = torch.nn.functional.cosine_similarity(h, hh, dim=-1).mean().item()
    # 负对:不同样本之间(h_i vs h_j, i≠j)的平均 cos
    sim = h @ h.t()  # [N,N]
    N = h.size(0)
    off_diag = sim[~torch.eye(N, dtype=torch.bool, device=h.device)]
    neg = off_diag.mean().item()
    return pos, neg


def train(z_all, z_hat_all, epochs=50, lr=1e-3, temperature=1.0,
          collapse_thresh=0.9):
    """核心训练循环。z_all, z_hat_all: [N, H] 已由冻结 BERT 编码好的句向量对。
    collapse_thresh: 负对 cos 超过此值视为坍缩,触发 early stop。"""
    H = z_all.size(1)
    adapter = DebiasAdapter(H).to(z_all.device)
    optim = Adam(adapter.parameters(), lr=lr)
    print(f"[优化器] 只优化 adapter, 参数量 = "
          f"{sum(p.numel() for p in adapter.parameters()):,}")

    pos0, neg0 = collapse_report(adapter, z_all, z_hat_all)
    print(f"[去偏前] 正对cos={pos0:.4f}  负对cos={neg0:.4f}")
    print(f"         (目标: 正对↑ 负对保持低; 若负对也↑到~1 = 坍缩)\n")

    best_adapter_state = None
    for ep in range(epochs):
        adapter.train()
        h = adapter(z_all)
        h_hat = adapter(z_hat_all)
        loss = info_nce_loss(h, h_hat, temperature)
        optim.zero_grad()
        loss.backward()
        optim.step()

        pos, neg = collapse_report(adapter, z_all, z_hat_all)
        if (ep + 1) % max(1, epochs // 10) == 0:
            flag = "  <- 坍缩!" if neg > collapse_thresh else ""
            print(f"  epoch {ep + 1:3d}  loss={loss.item():.4f}  "
                  f"正对={pos:.4f}  负对={neg:.4f}{flag}")

        # early stop: 负对 cos 冲过阈值,说明开始坍缩,停在坍缩前
        if neg > collapse_thresh:
            print(f"\n[early stop] 第 {ep + 1} 轮负对 cos={neg:.4f} > {collapse_thresh}, "
                  f"判定坍缩,停止训练")
            break
        best_adapter_state = {k: v.clone() for k, v in adapter.state_dict().items()}

    # 回滚到坍缩前的最后一个健康状态
    if best_adapter_state is not None:
        adapter.load_state_dict(best_adapter_state)

    pos1, neg1 = collapse_report(adapter, z_all, z_hat_all)
    print(f"\n[去偏后] 正对cos={pos1:.4f}  负对cos={neg1:.4f}")
    print(f"[判读] 正对 {pos0:.4f}->{pos1:.4f} (升高=性别差异被抹平)")
    print(f"       负对 {neg0:.4f}->{neg1:.4f} "
          f"({'健康,句子仍可区分 ✓' if neg1 < collapse_thresh else '偏高,仍有坍缩风险'})")
    return adapter


def test():
    torch.manual_seed(config.SEED)

    if "--fake" in sys.argv:
        print("[模式]假数据，仅验证训练循环坏逻辑")
        N, H = 12, 768
        base = torch.randn(N, H)
        z = base + 0.3 * torch.randn(N, H)
        z_hat = base + 0.3 * torch.randn(N, H)
        train(z, z_hat, epochs=50, lr=1e-3)
    else:
        # Mac 真实路径
        print(f"[模式] 真实 BERT, 设备 {config.DEVICE}\n")
        z, z_hat = build_pairs_with_bert()
        train(z, z_hat, epochs=50, lr=1e-3)


def main():
    test()


if __name__ == "__main__":
    main()
