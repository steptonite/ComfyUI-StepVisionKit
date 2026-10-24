"""Криві переносу й праймеріз для IO-нод: sRGB, BT.1886 (Rec.709 гамма 2.4), PQ (SMPTE 2084), HLG (BT.2100).

Домовленість паку: лінійне 1.0 = дифузний білий. У HDR воно лягає на REF_WHITE ніт
(BT.2408: 203 ніти для PQ і HLG 75 %), тож світліше за білий місце лишається для світлих ділянок.
"""
import numpy as np

REF_WHITE = 203.0

M709_TO_2020 = np.array([[0.6274040, 0.3292820, 0.0433136],
                         [0.0690970, 0.9195400, 0.0113612],
                         [0.0163916, 0.0880132, 0.8955950]], np.float32)
M2020_TO_709 = np.linalg.inv(M709_TO_2020).astype(np.float32)

_PQ_M1, _PQ_M2 = 2610 / 16384, 2523 / 4096 * 128
_PQ_C1, _PQ_C2, _PQ_C3 = 3424 / 4096, 2413 / 4096 * 32, 2392 / 4096 * 32
_HLG_A = 0.17883277
_HLG_B = 1 - 4 * _HLG_A
_HLG_C = 0.5 - _HLG_A * np.log(4 * _HLG_A)
HLG_REF_SCENE = 0.26496   # сцена-лінійне значення, що дає 75 % сигналу HLG (дифузний білий за BT.2408)


def srgb_oetf(x):
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def srgb_eotf(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4)).astype(np.float32)


def bt1886_inv(x):
    return np.power(np.clip(x, 0.0, 1.0), 1 / 2.4).astype(np.float32)


def bt1886(x):
    return np.power(np.clip(x, 0.0, 1.0), 2.4).astype(np.float32)


def pq_encode(lin, ref=REF_WHITE):
    y = np.clip(lin * (ref / 10000.0), 0.0, 1.0)
    p = np.power(y, _PQ_M1)
    return np.power((_PQ_C1 + _PQ_C2 * p) / (1 + _PQ_C3 * p), _PQ_M2).astype(np.float32)


def pq_decode(sig, ref=REF_WHITE):
    e = np.power(np.clip(sig, 0.0, 1.0), 1 / _PQ_M2)
    y = np.power(np.maximum(e - _PQ_C1, 0) / (_PQ_C2 - _PQ_C3 * e), 1 / _PQ_M1)
    return (y * (10000.0 / ref)).astype(np.float32)


def hlg_encode(lin):
    e = np.clip(lin * HLG_REF_SCENE, 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        hi = _HLG_A * np.log(np.maximum(12 * e - _HLG_B, 1e-12)) + _HLG_C
    return np.where(e <= 1 / 12, np.sqrt(3 * e), hi).astype(np.float32)


def hlg_decode(sig):
    s = np.clip(sig, 0.0, 1.0)
    e = np.where(s <= 0.5, s * s / 3, (np.exp((s - _HLG_C) / _HLG_A) + _HLG_B) / 12)
    return (e / HLG_REF_SCENE).astype(np.float32)


def apply_matrix(rgb, m):
    return np.einsum("...c,dc->...d", rgb, m).astype(np.float32)
