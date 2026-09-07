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
    return 0.0 if den == 0 or not np.isfinite(den) else float(num / den)

def bray_components(comm: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    x = comm.to_numpy(dtype=float)
    samples = list(comm.index)
    n = x.shape[0]
    mats = {k: np.zeros((n, n), dtype=float) for k in ["total", "balanced", "gradient"]}
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            a = float(np.minimum(x[i], x[j]).sum())
            b = float(np.maximum(x[i] - x[j], 0).sum())
            c = float(np.maximum(x[j] - x[i], 0).sum())
            den = 2 * a + b + c
            total = safe_div(b + c, den)
            balanced = safe_div(2 * min(b, c), den)
            gradient = safe_div(abs(b - c), den)
            vals = {"total": total, "balanced": balanced, "gradient": gradient}
            for key, val in vals.items():
                mats[key][i, j] = mats[key][j, i] = val
            rows.append(
                {
                    "sample_i": samples[i],
                    "sample_j": samples[j],
                    "shared_abundance_sum": a,
                    "only_i_abundance_sum": b,
                    "only_j_abundance_sum": c,
                    "beta_bray_total": total,
                    "beta_bray_balanced": balanced,
                    "beta_bray_gradient": gradient,
                    "similarity": max(0.0, 1.0 - total),
                    "note_zh": "Bray-Curtis 数量结构分解；balanced/gradient 不能写成 turnover/nestedness。",
                }
            )
    return mats, pd.DataFrame(rows)

def tree_abundance_components(comm: pd.DataFrame, membership: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    taxa = list(comm.columns)
    missing = sorted(set(taxa) - set(membership.columns))
    if missing:
        raise ValueError(f"树 membership 缺少这些分类单元: {missing}")
    edge_taxa = membership[taxa].to_numpy(dtype=float)
    weights = membership["branch_length"].to_numpy(dtype=float)
    keep = weights > 1e-12
    edge_taxa = edge_taxa[keep, :]
    weights = weights[keep]
    edge_ids = membership.loc[keep, "edge_id"].astype(str).tolist()
    branch_abund = comm.to_numpy(dtype=float) @ edge_taxa.T
    samples = list(comm.index)
    n = branch_abund.shape[0]
    mats = {k: np.zeros((n, n), dtype=float) for k in ["total", "replacement", "richness_difference"]}
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            xi = branch_abund[i]
            xj = branch_abund[j]
            a = float((weights * np.minimum(xi, xj)).sum())
            b = float((weights * np.maximum(xi - xj, 0)).sum())
            c = float((weights * np.maximum(xj - xi, 0)).sum())
            den = 2 * a + b + c
            total = safe_div(b + c, den)
            repl = safe_div(2 * min(b, c), den)
            rich = safe_div(abs(b - c), den)
            vals = {"total": total, "replacement": repl, "richness_difference": rich}
            for key, val in vals.items():
                mats[key][i, j] = mats[key][j, i] = val
            rows.append(
                {
                    "sample_i": samples[i],
                    "sample_j": samples[j],
                    "weighted_shared_branch_abundance": a,
                    "weighted_only_i_branch_abundance": b,
                    "weighted_only_j_branch_abundance": c,
                    "beta_total": total,
                    "beta_replacement": repl,
                    "beta_richness_difference": rich,
                    "similarity": max(0.0, 1.0 - total),
                    "n_positive_length_edges": len(edge_ids),
                    "note_zh": "基于树边后代分类单元的数量加权分解；作为补充分析，不等同于 incidence Baselga 嵌套分解。",
                }
            )
    return mats, pd.DataFrame(rows)

def dbrda_r2(Y: np.ndarray, X_pred: pd.DataFrame, n_perm: int = 499, rng: np.random.Generator | None = None) -> Tuple[float, float, float]:
    rng = rng or np.random.default_rng(RNG_SEED)
    y = np.asarray(Y, dtype=float)
    if y.ndim != 2 or y.size == 0:
        return np.nan, np.nan, np.nan
    y = y - np.nanmean(y, axis=0, keepdims=True)
    ss_total = float(np.nansum(y ** 2))
    if ss_total <= 0:
        return 0.0, 0.0, 1.0
    x = X_pred.to_numpy(dtype=float) if not X_pred.empty else np.zeros((y.shape[0], 0))
    if x.shape[1] == 0:
        return 0.0, 0.0, 1.0
    x = zscore_df(pd.DataFrame(x)).to_numpy(dtype=float)
    x = np.column_stack([np.ones(y.shape[0]), x])
    h = x @ np.linalg.pinv(x)
    fitted = h @ y
    r2 = float(np.nansum(fitted ** 2) / ss_total)
    p = x.shape[1] - 1
    n = y.shape[0]
    adj = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n - p - 1 > 0 else np.nan
    ge = 0
    for _ in range(n_perm):
        yp = y[rng.permutation(n), :]
        rp = float(np.nansum((h @ yp) ** 2) / ss_total)
        if rp >= r2 - 1e-15:
            ge += 1
    pval = (ge + 1) / (n_perm + 1)
    return r2, adj, pval
