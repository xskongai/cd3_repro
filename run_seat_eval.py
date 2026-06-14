#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kong Xiaoshuang
Date: 6/14/26
Description: run_seat_eval

SEAT 完整评测: 裸 BERT vs BERT+adapter
======================================
策略: 效应量绝对值在小词表上不稳(见 eval_seat.py 的验证),
所以我们看"去偏前后的相对变化"——adapter 若有效,效应量应更接近 0。

流程:
  1. 把 SEAT 词表套进 "This is a [word]" 模板
  2. 用冻结 BERT 编码 -> 词向量
  3. 算裸 BERT 的效应量
  4. 让词向量过 adapter -> 算去偏后效应量
  5. 对比

"""

import numpy as np
import torch

import config
from eval_seat import (MALE_WORDS, FEMALE_WORDS, CAREER_WORDS, FAMILY_WORDS, SEAT_TEMPLATE, effect_size)

from stage1_search import encode
from stage2_train import build_pairs_with_bert, train


def embed_word(words):
    """把词套模板、用冻结 BERT 编码成向量, 返回 numpy [n,H]。"""
    sents = [SEAT_TEMPLATE.format(w) for w in words]
    vecs = encode(sents)
    return vecs.detach().cpu().numpy()


def run_seat(adapter=None):
    """跑一次 SEAT。adapter=None 时测裸 BERT;否则测去偏后。"""

    def emb(words):
        v = embed_word(words)
        if adapter is not None:
            with torch.no_grad():
                t = torch.tensor(v, device=config.DEVICE)
                v = adapter(t).detach().cpu().numpy()
        return v

    X, Y = emb(MALE_WORDS), emb(FEMALE_WORDS)
    A, B = emb(CAREER_WORDS), emb(FAMILY_WORDS)
    return effect_size(X, Y, A, B)


def test():
    torch.manual_seed(config.SEED)
    print("=" * 55)
    print("SEAT 评测: gender × (career vs family)")
    print("效应量越接近 0 越公平。绝对值有噪声,重点看变化。")
    print("=" * 55)
    # 1. BERT
    es_before = run_seat(adapter=None)
    print(f"\n[裸 BERT]      effect size = {es_before:+.4f}")

    # 2. 训练 adapter
    print("\n--- 训练去偏 adapter ---")
    z, z_hat = build_pairs_with_bert()
    adapter = train(z, z_hat, epochs=50, lr=1e-3)

    # 3. 去偏后
    es_after = run_seat(adapter=adapter)
    print(f"\n[BERT+adapter] effect size = {es_after:+.4f}")

    # 4. 判读
    print("\n" + "=" * 55)
    print(f"效应量 |{es_before:+.4f}| -> |{es_after:+.4f}|")
    if abs(es_after) < abs(es_before):
        print("去偏后更接近 0 -> adapter 起作用了 ✓")
    else:
        print("去偏后未更接近 0 -> adapter 没泛化到 SEAT 词(见下方分析)")
    print("=" * 55)


def main():
    test()


if __name__ == "__main__":
    main()
