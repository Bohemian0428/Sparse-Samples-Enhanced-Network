# -*- coding: utf-8 -*-
"""数据预处理：计算光谱指数，并划分训练 / 验证 / 测试集。

输入 (N, H, W, 7)：0-5 为 SR_B2~SR_B7，6 为标签（0-5，负值无效）
输出 X (n, H, W, 12) float32，Y (n, H, W) int64
"""
import glob
import os

import numpy as np
from sklearn.model_selection import train_test_split

RAW_BANDS = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
INDEX_POOL = ['NDVI', 'NDMI', 'NDWI', 'RVI', 'ARVI', 'EVI']
DEFAULT_BANDS = RAW_BANDS + INDEX_POOL + ['Landcover']

LABEL_BAND = 'Landcover'
NUM_CLASSES = 6


def load_array(path, part_glob=None):
    if path and os.path.exists(path):
        return np.load(path)
    if part_glob:
        parts = sorted(glob.glob(part_glob))
        if parts:
            return np.concatenate([np.load(p) for p in parts], axis=0)
    raise FileNotFoundError('未找到数据文件：%s' % path)


def compute_indices(data, names):
    b2, b3, b4 = data[..., 0], data[..., 1], data[..., 2]
    b5, b6 = data[..., 3], data[..., 4]

    out = []
    for nm in names:
        with np.errstate(divide='ignore', invalid='ignore'):
            if nm == 'NDVI':
                v = (b5 - b4) / (b5 + b4)
            elif nm == 'NDMI':
                v = (b5 - b6) / (b5 + b6)
            elif nm == 'NDWI':
                v = (b3 - b5) / (b3 + b5)
            elif nm == 'RVI':
                v = b5 / b4
            elif nm == 'ARVI':
                v = (b5 - (b4 - b2)) / (b5 + (b4 - b2))
            elif nm == 'EVI':
                v = 2.5 * (b5 - b4) / (b5 + 6 * b4 - 7.5 * b2 + 1)
            else:
                raise ValueError('未知指数：%s' % nm)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        out.append(v[..., np.newaxis])
    return out


def build_matrix(data, bands=None):
    bands = list(bands or DEFAULT_BANDS)
    if bands[-1] != LABEL_BAND:
        bands.append(LABEL_BAND)
    feats = bands[:-1]

    if [b for b in feats if b in RAW_BANDS] != RAW_BANDS:
        raise ValueError('原始波段需按顺序完整给出：%s' % RAW_BANDS)
    if data.ndim != 4 or data.shape[-1] != 7:
        raise ValueError('输入应为 (N, H, W, 7)，实际 %s' % (data.shape,))

    idx = [b for b in feats if b not in RAW_BANDS]
    blocks = [data[..., :6].astype(np.float32)]
    blocks += compute_indices(data, idx)
    blocks.append(data[..., 6:7].astype(np.float32))
    return np.concatenate(blocks, axis=-1)


def split_index(n, val_size=0.3, test_size=0.3, seed=42):
    idx = np.arange(n)
    hold = val_size + test_size
    if not 0 < hold < 1:
        raise ValueError('val_size + test_size 需在 (0, 1) 内')
    i_tr, i_hold = train_test_split(idx, test_size=hold, random_state=seed)
    i_va, i_te = train_test_split(i_hold, test_size=test_size / hold, random_state=seed)
    return np.sort(i_tr), np.sort(i_va), np.sort(i_te)


def load_and_split(path=None, bands=None, val_size=0.3, test_size=0.3, seed=42,
                   part_glob=None):
    data = load_array(path, part_glob)
    M = build_matrix(data, bands)

    X = np.ascontiguousarray(M[..., :-1])
    Y = np.nan_to_num(M[..., -1], nan=-1.0).astype(np.int64)
    Y[(Y < 0) | (Y >= NUM_CLASSES)] = -1

    i_tr, i_va, i_te = split_index(X.shape[0], val_size, test_size, seed)
    return (X[i_tr], Y[i_tr], X[i_va], Y[i_va], X[i_te], Y[i_te])
