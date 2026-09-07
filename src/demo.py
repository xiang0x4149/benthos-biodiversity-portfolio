"""从零生成示例群落；真实性状仅用于说明分析接口，不重现论文结论。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from metrics import decompose, weighted_gower, functional_tree, taxonomy_tree, paired_tests, bh_adjust

ROOT = Path(__file__).resolve().parents[1]


def run():
    trait_dir = ROOT / "data" / "traits"
    demo_dir = ROOT / "data" / "synthetic"
    out = ROOT / "outputs" / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
    taxa = pd.read_csv(trait_dir / "taxa.csv", keep_default_na=False)
    traits = pd.read_csv(trait_dir / "trait_matrix.csv").set_index("taxon_id")
    schema = pd.read_csv(trait_dir / "trait_schema.csv").set_index("output_column")
    ids = taxa.taxon_id.tolist()
    traits = traits.loc[ids]
    rng = np.random.default_rng(20260907)
    # 24 个虚构样本，数值分布固定写在代码里，没有任何真实群落数据入口。
    samples = [f"DEMO_{s}_{i:02d}" for s in ("summer", "autumn") for i in range(1, 13)]
    mapping = []
    for i in range(47):
        target = ids[i] if i < 37 else ids[i-37] if i < 43 else ""
        mapping.append({"record_id": f"DEMO_R{i+1:03d}", "synthetic_name": f"虚构原始条目{i+1:03d}",
                        "taxon_id": target, "action": "merge_sum" if 37 <= i < 43 else "retain" if target else "exclude",
                        "reason_zh": "模拟分类层级不足而剔除" if not target else "演示合并规则；不是历史真实异名映射"})
    mapping = pd.DataFrame(mapping)
    mapping.to_csv(demo_dir / "mapping.csv", index=False)
    rows = []
    for j, sample in enumerate(samples):
        p = rng.uniform(.08, .3)
        density = rng.integers(1, 30, 47) * (rng.random(47) < p)
        # 保证每个演示样本至少有 2 个保留类群，不从真实丰富度拟合。
        density[:2] += rng.integers(1, 4, 2)
        mass = np.round(density * rng.uniform(.01, 4, 47), 4)
        for i in range(47):
            rows.append({"sample_id": sample, "record_id": mapping.iloc[i].record_id,
                         "density": int(density[i]), "biomass": float(mass[i])})
    raw = pd.DataFrame(rows)
    raw.to_csv(demo_dir / "observations.csv", index=False)
    metadata = pd.DataFrame({"sample_id": samples, "site_id": [f"DEMO_SITE_{i:02d}" for i in range(1,13)] * 2,
                             "season": ["summer"]*12 + ["autumn"]*12,
                             "data_origin": ["synthetic_only"]*24})
    metadata.to_csv(demo_dir / "metadata.csv", index=False)
    valid = raw.merge(mapping[["record_id", "taxon_id"]], on="record_id", validate="many_to_one")
    valid = valid[valid.taxon_id != ""]
    grouped = valid.groupby(["sample_id", "taxon_id"], sort=False)[["density", "biomass"]].sum()
    comm = {kind: grouped[kind].unstack().reindex(index=samples, columns=ids).fillna(0).to_numpy()
            for kind in ("density", "biomass")}
    comm["incidence"] = (comm["density"] > 0).astype(int)
    # 密度和生物量合并守恒；有无值必须重新二值化。
    for kind in ("density", "biomass"):
        assert np.isclose(comm[kind].sum(), valid[kind].sum())
    processed = grouped.reset_index()
    processed["incidence"] = (processed.density > 0).astype(int)
    processed.to_csv(out / "community_long.csv", index=False)
    gower, audit, weights = weighted_gower(traits.to_numpy(), schema.loc[traits.columns, "standard_trait_group_cn"].tolist())
    func_mem, func_len = functional_tree(gower)
    tax_mem, tax_len = taxonomy_tree(taxa.to_dict("records"))
    summary, season_tests = [], []
    for kind in ("incidence", "density", "biomass"):
        for dim in ("TD", "FD", "PD_proxy"):
            values = comm[kind]
            edge_weights = None
            if dim != "TD":
                mem, edge_weights = (func_mem, func_len) if dim == "FD" else (tax_mem, tax_len)
                values = values @ mem.T
                if kind == "incidence":values = (values > 0).astype(int)
            partition = "baselga" if kind == "incidence" and dim != "FD" else "podani"
            coefficient = "jaccard" if kind == "incidence" and dim == "FD" else "sorensen"
            total, first, second = decompose(values, coefficient=coefficient, partition=partition, weights=edge_weights)
            if partition == "baselga":a, b = "turnover", "nestedness_resultant"
            elif dim == "TD":a, b = "balanced_variation", "abundance_gradient"
            else:a, b = "replacement", "richness_difference" if kind == "incidence" else "abundance_difference"
            for season, idx in [("summer", np.arange(12)), ("autumn", np.arange(12,24))]:
                tri = np.triu_indices(12, 1)
                nums = [float(m[np.ix_(idx,idx)][tri].mean()) for m in (total,first,second)]
                assert np.isclose(nums[0], nums[1]+nums[2])
                summary.append({"data_type":kind,"dimension":dim,"season":season,"coefficient":coefficient,
                                "partition":partition,"component_a":a,"component_b":b,"n_pairs":66,
                                "mean_total":nums[0],"mean_component_a":nums[1],"mean_component_b":nums[2]})
            result = paired_tests(total, 12)
            season_tests.append({"data_type":kind,"dimension":dim,"coefficient":coefficient,"partition":partition,**result})
    pd.DataFrame(summary).to_csv(out / "beta_summary.csv", index=False)
    test_table = pd.DataFrame(season_tests)
    test_table["q_permanova"] = bh_adjust(test_table.p_permanova)
    test_table["q_permdisp"] = bh_adjust(test_table.p_permdisp)
    test_table.to_csv(out / "season_tests.csv", index=False)
    metrics = {"data_origin_zh":"全部群落观测为从零生成的合成数据；不拟合真实值或真实分布",
               "seed":20260907,"raw_records":47,"retained_taxa":37,"sample_count":24,
               "no_private_input_read":True,"gower":audit,
               "statistical_scope_zh":"九个总差异矩阵；同点受限精确枚举 4096 标签排列；两项检验均采用明确的 Lingoes 校正；BH 在九个响应内分别校正",
               "not_reproduced_zh":"没有重现真实样本、论文 p 值、环境驱动、TBI 或四维凸包敏感性分析"}
    (out / "quality_checks.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2)+"\n")
    print("合成演示完成：47 条虚构原始记录 → 37 个分析单元；24 个虚构样本；18 行季节摘要、9 个季节检验。")


if __name__ == "__main__":run()
