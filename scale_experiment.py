"""
规模扫描实验: 训练语料规模 vs SEAT 效应量
=========================================
对照实验,验证我们的发现:"小规模过拟合不泛化,加数据应改善泛化"。

做法: 对每个语料规模 N,生成 N 句 -> 反事实对 -> 冻结BERT编码 -> 训adapter
      -> 测 SEAT 效应量。并排打印,看效应量是否随 N 向 0 靠拢。

这是一个 ablation 雏形(和论文 5.3 同构)。重点看趋势,不是绝对值。
"""

import numpy as np
import torch

import config
from corpus_gen import generate_sentences
from stage1_augment import make_counterfactual_pair
from stage1_search import encode
from stage2_train import train


def encode_pairs(sentences):
    """句子 -> 反事实句对 -> 冻结BERT编码成 (z, z_hat)。"""
    xs, xhs = [], []
    for s in sentences:
        x, xh, _ = make_counterfactual_pair(s)
        if xh:
            xs.append(x); xhs.append(xh)
    z = encode(xs).detach()
    z_hat = encode(xhs).detach()
    return z, z_hat


def run_seat_with_adapter(adapter):
    """复用 SEAT 词表,测一次效应量。"""
    from eval_seat import (MALE_WORDS, FEMALE_WORDS, CAREER_WORDS,
                           FAMILY_WORDS, SEAT_TEMPLATE, effect_size)

    def emb(words):
        sents = [SEAT_TEMPLATE.format(w) for w in words]
        v = encode(sents)
        if adapter is not None:
            with torch.no_grad():
                v = adapter(v)
        return v.detach().cpu().numpy()

    return effect_size(emb(MALE_WORDS), emb(FEMALE_WORDS),
                       emb(CAREER_WORDS), emb(FAMILY_WORDS))


if __name__ == "__main__":
    torch.manual_seed(config.SEED)

    # 先测裸 BERT 作基线
    es_bare = run_seat_with_adapter(adapter=None)
    print("=" * 55)
    print(f"基线 [裸 BERT] effect size = {es_bare:+.4f}")
    print("=" * 55)

    SIZES = [12, 40, 100, 200]
    results = []
    for n in SIZES:
        print(f"\n### 语料规模 N = {n} ###")
        sents = generate_sentences(n)
        z, z_hat = encode_pairs(sents)
        print(f"  实际句对数: {z.size(0)}")
        adapter = train(z, z_hat, epochs=50, lr=1e-3)
        es = run_seat_with_adapter(adapter)
        results.append((n, es))
        print(f"  -> N={n}: SEAT effect size = {es:+.4f}  (|{abs(es):.4f}|)")

    # 汇总趋势
    print("\n" + "=" * 55)
    print("规模扫描汇总 (效应量绝对值越小越公平):")
    print(f"  裸 BERT          |{abs(es_bare):.4f}|")
    for n, es in results:
        print(f"  N={n:<4d}          |{abs(es):.4f}|   ({es:+.4f})")
    print("=" * 55)
    print("看 |effect size| 是否随 N 增大而减小 -> 验证'规模改善泛化'")