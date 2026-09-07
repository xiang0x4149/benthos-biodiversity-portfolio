"""历史方法函数摘录：仅供代码审阅；与公开演示的整理实现分开。
提取自 2026-05-20 前期分析，不包含任何数据读取、写入和批处理入口。
主要由 AI/agent 生成；历史实现中的近似或限制不因归档而获得验证。
"""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
from typing import Dict, Iterable, Sequence, Tuple

def safe_div(num: float, den: float) -> float:
    if den == 0 or not np.isfinite(den):
        return 0.0
    return float(num / den)

def component_values(a: float, b: float, c: float) -> Dict[str, float]:
    min_bc = min(b, c)
    sor = safe_div(b + c, 2 * a + b + c)
    sim = safe_div(min_bc, a + min_bc)
    sne = max(0.0, sor - sim)
    jac = safe_div(b + c, a + b + c)
    jtu = safe_div(2 * min_bc, a + 2 * min_bc)
    jne = max(0.0, jac - jtu)
    repl = safe_div(2 * min_bc, a + b + c)
    rich = safe_div(abs(b - c), a + b + c)
    hill_eff = 1.0 + sor
    return {
        "sor_total": sor,
        "sor_turnover": sim,
        "sor_nestedness": sne,
        "jaccard_total": jac,
        "jaccard_turnover": jtu,
        "jaccard_nestedness": jne,
        "podani_replacement": repl,
        "podani_richness": rich,
        "hill_q0_effective_beta": hill_eff,
        "hill_q0_differentiation": sor,
    }

def pairwise_abc(binary: np.ndarray, weights: np.ndarray | None = None) -> Dict[str, np.ndarray]:
    x = np.asarray(binary, dtype=float)
    n = x.shape[0]
    w = np.ones(x.shape[1], dtype=float) if weights is None else np.asarray(weights, dtype=float)
    a = np.zeros((n, n), dtype=float)
    b = np.zeros((n, n), dtype=float)
    c = np.zeros((n, n), dtype=float)
    for i in range(n):
        xi = x[i] > 0
        for j in range(i + 1, n):
            xj = x[j] > 0
            shared = float(w[xi & xj].sum())
            only_i = float(w[xi & ~xj].sum())
            only_j = float(w[~xi & xj].sum())
            a[i, j] = a[j, i] = shared
            b[i, j] = c[j, i] = only_i
            c[i, j] = b[j, i] = only_j
    return {"a": a, "b": b, "c": c}

def multiple_site_summary(binary: np.ndarray) -> Dict[str, float]:
    x = np.asarray(binary, dtype=float)
    shared = x @ x.T
    site_richness = np.diag(shared)
    not_shared = np.abs(shared - site_richness.reshape(1, -1))
    sum_si = float(site_richness.sum())
    st = float((x.sum(axis=0) > 0).sum())
    a_multi = sum_si - st
    maxbibj = float(np.maximum(not_shared, not_shared.T)[np.tril_indices_from(not_shared, k=-1)].sum())
    minbibj = float(np.minimum(not_shared, not_shared.T)[np.tril_indices_from(not_shared, k=-1)].sum())
    beta_sim = safe_div(minbibj, minbibj + a_multi)
    beta_sne = safe_div(a_multi, minbibj + a_multi) * safe_div(maxbibj - minbibj, 2 * a_multi + maxbibj + minbibj)
    beta_sor = safe_div(minbibj + maxbibj, minbibj + maxbibj + 2 * a_multi)
    beta_jtu = safe_div(2 * minbibj, 2 * minbibj + a_multi)
    beta_jne = safe_div(a_multi, 2 * minbibj + a_multi) * safe_div(maxbibj - minbibj, a_multi + maxbibj + minbibj)
    beta_jac = safe_div(minbibj + maxbibj, minbibj + maxbibj + a_multi)
    return {
        "sum_site_richness": sum_si,
        "regional_richness": st,
        "a_multi": a_multi,
        "sum_min_not_shared": minbibj,
        "sum_max_not_shared": maxbibj,
        "beta_SIM_turnover": beta_sim,
        "beta_SNE_nestedness": beta_sne,
        "beta_SOR_total": beta_sor,
        "beta_JTU_turnover": beta_jtu,
        "beta_JNE_nestedness": beta_jne,
        "beta_JAC_total": beta_jac,
    }
