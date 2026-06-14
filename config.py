"""
CD3 复现 - 集中配置
====================
所有可调旋钮集中放这里。想往论文设置逼近时,只改这个文件,别动逻辑代码。

论文原始设置 (5.1.5) vs 我们的 Mac 缩小版:
  词表大小 |V|   : 论文 5000   -> 我们先 500
  beam 宽度 K   : 论文 20     -> 我们先 10
  迭代轮数 ε    : 论文 5      -> 我们先 3
  句对数        : 论文 269    -> 我们先 15 (用内置 demo 句)
  设备          : 论文 4xGPU  -> 我们 MPS

机制完全一致,只是规模缩了 —— 你能看到"搜出的词确实偏",而不是复刻 6 小时。
"""

import torch

# ---------- 设备 ----------
# Mac 走 MPS;没有就退回 CPU
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# ---------- 模型 ----------
MODEL_NAME = "bert-base-uncased"   # 论文三个 PLM 之一,先用 BERT

# ---------- Stage1 Step2: biased prompt 搜索 ----------
VOCAB_SIZE = 500     # 候选词表大小 (论文 5000)。Mac 上先小,跑顺了再加大
BEAM_K = 10          # beam 宽度 (论文 20)
NUM_ITERS = 3        # 迭代轮数 (论文 5)
MAX_PROMPT_PER_ITER_PRINT = 8   # 每轮打印多少个搜到的 prompt,纯展示用

# ---------- 句向量池化方式 ----------
# "mean": 所有 token 平均(论文用法,但会稀释单敏感词信号)
# "cls" : 用 [CLS] 向量
# "diff": 只取敏感词位置的向量(信号最锐利,不被稀释)—— 实验用
POOLING = "mean"

# ---------- 可复现性 ----------
SEED = 42