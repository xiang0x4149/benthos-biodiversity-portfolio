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

def components_from_feature_values(values: np.ndarray, weights: np.ndarray | None = None, abundance: bool = False) -> Dict[str, np.ndarray]:
    x = np.asarray(values, dtype=float)
    n, p = x.shape
    w = np.ones(p, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    out = {k: np.zeros((n, n), dtype=float) for k in ["total", "replacement", "richness_difference", "baselga_turnover", "baselga_nestedness"]}
    for i in range(n):
        xi = x[i]
        for j in range(i + 1, n):
            xj = x[j]
            if abundance:
                a = float((w * np.minimum(xi, xj)).sum())
                b = float((w * np.maximum(xi - xj, 0)).sum())
                c = float((w * np.maximum(xj - xi, 0)).sum())
            else:
                pi = xi > 0
                pj = xj > 0
                a = float(w[pi & pj].sum())
                b = float(w[pi & ~pj].sum())
                c = float(w[~pi & pj].sum())
            den = 2 * a + b + c
            total = safe_div(b + c, den)
            replacement = safe_div(2 * min(b, c), den)
            richness = safe_div(abs(b - c), den)
            turnover = safe_div(min(b, c), a + min(b, c))
            nestedness = max(0.0, total - turnover)
            vals = {
                "total": total,
                "replacement": replacement,
                "richness_difference": richness,
                "baselga_turnover": turnover,
                "baselga_nestedness": nestedness,
            }
            for key, val in vals.items():
                out[key][i, j] = out[key][j, i] = val
    return out

def gower_center(dist: np.ndarray) -> np.ndarray:
    d = np.asarray(dist, dtype=float)
    n = d.shape[0]
    j = np.eye(n) - np.ones((n, n)) / n
    return -0.5 * j @ (d ** 2) @ j

def design_matrix(labels: Sequence[str]) -> np.ndarray:
    cats = sorted(pd.Series(labels).astype(str).unique().tolist())
    return np.column_stack([(pd.Series(labels).astype(str).to_numpy() == c).astype(float) for c in cats])

def permanova_f(dist: pd.DataFrame, labels: Sequence[str]) -> Tuple[float, float, int, int]:
    arr = dist.to_numpy(dtype=float)
    a = gower_center(arr)
    labels_arr = np.asarray(labels, dtype=object)
    groups = [np.where(labels_arr == g)[0] for g in pd.unique(labels_arr)]
    ss_between = float(sum(a[np.ix_(idx, idx)].sum() / len(idx) for idx in groups if len(idx) > 0))
    ss_total = float(np.trace(a))
    ss_within = max(0.0, ss_total - ss_between)
    g = len(groups)
    n = len(labels)
    df_between = g - 1
    df_within = n - g
    fval = safe_div(ss_between / max(df_between, 1), ss_within / max(df_within, 1))
    r2 = safe_div(ss_between, ss_total)
    return fval, r2, df_between, df_within

def permuted_label_arrays(labels: Sequence[str], strata: Sequence[str] | None, n_perm: int, rng: np.random.Generator) -> Iterable[np.ndarray]:
    labels_arr = np.asarray(labels, dtype=object)
    if strata is not None:
        strata_arr = np.asarray(strata, dtype=object)
        groups = [np.where(strata_arr == s)[0] for s in pd.unique(strata_arr)]
        if all(len(g) == 2 for g in groups) and len(set(labels_arr)) == 2:
            for flips in itertools.product([0, 1], repeat=len(groups)):
                perm = labels_arr.copy()
                for flip, idx in zip(flips, groups):
                    if flip:
                        perm[idx] = perm[idx[::-1]]
                yield perm
            return
        for _ in range(n_perm):
            perm = labels_arr.copy()
            for idx in groups:
                perm[idx] = rng.permutation(perm[idx])
            yield perm
    else:
        for _ in range(n_perm):
            yield rng.permutation(labels_arr)

def permanova_test(dist: pd.DataFrame, metadata: pd.DataFrame, group_col: str, strata_col: str | None, n_perm: int, rng: np.random.Generator) -> Dict[str, object]:
    meta = metadata.set_index("sample_id").loc[list(dist.index)]
    labels = meta[group_col].astype(str).to_numpy()
    strata = meta[strata_col].astype(str).to_numpy() if strata_col else None
    f_obs, r2, dfb, dfw = permanova_f(dist, labels)
    ge = 0
    n_actual = 0
    for perm in permuted_label_arrays(labels, strata, n_perm, rng):
        f_perm, _, _, _ = permanova_f(dist, perm)
        ge += int(f_perm >= f_obs - 1e-15)
        n_actual += 1
    return {
        "group_col": group_col,
        "strata_col": strata_col or "",
        "pseudo_F": f_obs,
        "R2": r2,
        "df_between": dfb,
        "df_within": dfw,
        "p_value": safe_div(ge + 1, n_actual + 1),
        "n_permutations": n_actual,
    }

def bh_fdr(p_values: Sequence[float]) -> List[float]:
    p = np.asarray(p_values, dtype=float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out.tolist()
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out[order] = np.clip(ranked, 0, 1)
    return out.tolist()
