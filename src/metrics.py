"""公开演示的计算核心：明确区分系数、组分、缺失假设与树代理。

根据毕业论文历史方法实现重新整理；主要由 AI 辅助实现。
此模块只接收内存数组，不读取研究工作区，也不生成论文真实样本。
"""
from __future__ import annotations

import itertools
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def decompose(values, *, coefficient="sorensen", partition="podani", weights=None):
    """返回总差异、第一组分、第二组分；输入必须为非负有限数。

    podani：第一项为替换/平衡变化，第二项为丰富度/数量差异。
    baselga：仅适用于有无输入，第一项为周转，第二项为嵌套结果。
    是否加权分支由调用者明确指定，避免把树代理误称为分子树。
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("群落矩阵必须为二维、有限、非负数组")
    if coefficient not in {"sorensen", "jaccard"} or partition not in {"podani", "baselga"}:
        raise ValueError("不支持的距离系数或分解框架")
    if partition == "baselga" and not np.isin(x, [0, 1]).all():
        raise ValueError("本实现的 Baselga 路线仅接受有无数据")
    w = np.ones(x.shape[1]) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (x.shape[1],) or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("特征权重不合法")
    a = (np.minimum(x[:, None, :], x[None, :, :]) * w).sum(axis=2)
    b = (np.maximum(x[:, None, :] - x[None, :, :], 0) * w).sum(axis=2)
    c = b.T
    den = (2 if coefficient == "sorensen" else 1) * a + b + c
    if np.any(np.diag(den) == 0):
        raise ValueError("存在空群落；请先确认样本定义，不能静默生成距离")
    total = np.divide(b + c, den, out=np.zeros_like(a), where=den > 0)
    if partition == "podani":
        first = np.divide(2 * np.minimum(b, c), den, out=np.zeros_like(a), where=den > 0)
    else:
        m = np.minimum(b, c)
        k = 1 if coefficient == "sorensen" else 2
        first = np.divide(k * m, a + k * m, out=np.zeros_like(a), where=(a + k * m) > 0)
    second = total - first
    return total, first, second


def weighted_gower(traits, groups):
    """性状块覆盖度开平方加权；成对剔除缺失。未定义距离明确按最大值补。

    编码列使用当前观测量程。返回距离、审计指标、最终列权重。
    这是历史主线的一种明确化实现，不代表所有 Gower 变体。
    """
    x = np.asarray(traits, dtype=float)
    if x.ndim != 2 or len(groups) != x.shape[1]:
        raise ValueError("性状与分组列数不一致")
    coverage = np.isfinite(x).mean(axis=0)
    groups = np.asarray(groups)
    w = np.zeros(x.shape[1])
    for g in np.unique(groups):
        mask = groups == g
        if coverage[mask].sum() > 0:
            w[mask] = np.sqrt(coverage[mask].mean()) * coverage[mask] / coverage[mask].sum()
    if w.sum() == 0:
        raise ValueError("所有性状均缺失")
    w /= w.sum()
    span = np.array([np.ptp(col[np.isfinite(col)]) if np.isfinite(col).any() else 0 for col in x.T])
    n = len(x)
    d = np.zeros((n, n))
    undefined, zeros = [], []
    for i in range(n):
        for j in range(i + 1, n):
            ok = np.isfinite(x[i]) & np.isfinite(x[j]) & (span > 0) & (w > 0)
            if not ok.any():
                value = np.nan
                undefined.append([i, j])
            else:
                value = np.average(np.abs(x[i, ok] - x[j, ok]) / span[ok], weights=w[ok])
                if value < 1e-12:
                    zeros.append([i, j])
            d[i, j] = d[j, i] = value
    valid = d[np.triu_indices(n, 1)]
    valid = valid[np.isfinite(valid)]
    if len(valid) == 0:
        raise ValueError("没有任何可比较的性状对")
    fill = float(valid.max())
    d = np.where(np.isnan(d), fill, d)
    audit = {"undefined_pair_count": len(undefined), "undefined_pair_indices": undefined,
             "zero_pair_count": len(zeros), "fill_value": fill,
             "missing_rule_zh": "成对剔除缺失；无共同有效性状的距离按当前可得最大值填补，属于明确假设"}
    return d, audit, w


def functional_tree(distance):
    """UPGMA 功能树：返回分支×末端成员矩阵及枝长，不是分子系统发育树。"""
    d = np.asarray(distance)
    z = linkage(squareform(d, checks=True), method="average")
    n = len(d)
    members = {i: np.eye(n)[i] for i in range(n)}
    heights = {i: 0.0 for i in range(n)}
    edges, lengths = [], []
    for k, (a, b, dist, _) in enumerate(z):
        node = n + k
        heights[node] = float(dist) / 2
        members[node] = members[int(a)] + members[int(b)]
        for child in (int(a), int(b)):
            length = max(0.0, heights[node] - heights[child])
            if length > 1e-12:
                edges.append(members[child]); lengths.append(length)
    return np.asarray(edges), np.asarray(lengths)


def taxonomy_tree(taxa):
    """分类阶元路径构建单位枝长代理树；跳过空阶元，保留末端边。"""
    ranks = ["kingdom", "phylum", "class", "order", "family", "genus"]
    paths = {}
    for i, row in enumerate(taxa):
        path = []
        for rank in ranks:
            if row.get(rank):
                path.append(rank + ":" + row[rank])
                paths.setdefault(tuple(path), set()).add(i)
        paths.setdefault(tuple(path + ["tip:" + row["taxon_id"]]), set()).add(i)
    mem = np.zeros((len(paths), len(taxa)))
    for j, indices in enumerate(paths.values()):
        mem[j, list(indices)] = 1
    return mem, np.ones(len(paths))


def lingoes_coordinates(distance):
    """存在负特征值时应用 Lingoes 欧氏化；返回坐标与校正常数。"""
    d = np.asarray(distance, float)
    n = len(d); h = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * h @ d**2 @ h
    vals = np.linalg.eigvalsh(b)
    constant = max(0.0, -float(vals.min()))
    if constant > 1e-10:
        d = np.sqrt(np.maximum(d**2 + 2 * constant, 0))
        np.fill_diagonal(d, 0)
        b = -0.5 * h @ d**2 @ h
    vals, vecs = np.linalg.eigh(b)
    keep = vals > 1e-10
    return vecs[:, keep] * np.sqrt(vals[keep]), constant


def paired_tests(distance, n_sites):
    """两季按相同固定点顺序排列；枚举 2^n_sites 个受限标签置换。

    为明确处理非欧氏分量，两项检验均用 Lingoes 校正距离表示。
    精确枚举包含观测排列一次，p=count(F_perm>=F_obs)/排列总数。
    此公共演示的明确校正流程不宣称复现历史论文 p 值。
    """
    if len(distance) != 2 * n_sites or n_sites < 2 or n_sites > 16:
        raise ValueError("受限置换要求两季配对输入且固定点数在 2 到 16 之间")
    coords, constant = lingoes_coordinates(distance)
    signs = np.array(list(itertools.product([-1., 1.], repeat=n_sites)))
    labels = np.hstack([signs, -signs])
    n = 2 * n_sites
    total_ss = float((coords**2).sum())
    if total_ss < 1e-12:
        return {"R2": 0., "pseudo_F": 0., "p_permanova": 1., "p_permdisp": 1.,
                "n_permutations": len(labels), "lingoes_constant": constant}
    centered_sum = labels @ coords
    between = (centered_sum**2).sum(axis=1) / n
    f = between / np.maximum(total_ss - between, 1e-15) * (n - 2)
    # 两组大小相同，整体坐标中心为零，各组中心由符号加权和确定。
    delta = centered_sum / n
    dist_to_center = np.linalg.norm(coords[None, :, :] - labels[:, :, None] * delta[:, None, :], axis=2)
    grand = dist_to_center.mean(axis=1)
    diff = (dist_to_center * labels).sum(axis=1) / n
    disp_between = n * diff**2
    disp_total = ((dist_to_center - grand[:, None])**2).sum(axis=1)
    disp_f = disp_between / np.maximum(disp_total - disp_between, 1e-15) * (n - 2)
    return {"R2": float(between[0] / total_ss), "pseudo_F": float(f[0]),
            "p_permanova": float(np.mean(f >= f[0] - 1e-10)),
            "p_permdisp": float(np.mean(disp_f >= disp_f[0] - 1e-10)),
            "n_permutations": len(labels), "lingoes_constant": constant}


def bh_adjust(p):
    p = np.asarray(p, float)
    order = np.argsort(p); ranked = p[order] * len(p) / np.arange(1, len(p)+1)
    result = np.empty_like(p); result[order] = np.minimum(1, np.minimum.accumulate(ranked[::-1])[::-1])
    return result
