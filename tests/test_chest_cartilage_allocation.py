from pathlib import Path
import sys
import types

import pandas as pd


APP_PATH = Path(r"F:/llqdocument/new modle/app_streamlit.py")
RATIO_PATH = Path(r"F:/llqdocument/大成文件/胸软骨占比20260526.xlsx")


def load_calculation_namespace():
    source = APP_PATH.read_text(encoding="utf-8-sig")
    cutoff = source.index('st.sidebar.header("数据源（请上传）")')
    module = types.ModuleType("app_streamlit_calculation")
    streamlit_stub = types.SimpleNamespace(
        sidebar=types.SimpleNamespace(file_uploader=lambda *args, **kwargs: None),
        cache_data=lambda *args, **kwargs: (lambda fn: fn),
        cache_resource=lambda *args, **kwargs: (lambda fn: fn),
        error=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        dataframe=lambda *args, **kwargs: None,
    )
    sys.modules.setdefault("streamlit", streamlit_stub)
    exec(compile(source[:cutoff], str(APP_PATH), "exec"), module.__dict__)
    return module.__dict__


def test_chest_cartilage_ratio_workbook_becomes_structure_ratio_rules():
    ns = load_calculation_namespace()

    rules = ns["read_chest_cartilage_structure_ratio"](str(RATIO_PATH))

    dalian = rules[rules["物料号"] == "ADG0183380610"].iloc[0]
    assert dalian["来源"] == "大连"
    assert round(float(dalian["脖子占比"]), 6) == 0.231874
    assert round(float(dalian["里肌占比"]), 6) == 0.178865
    assert round(float(dalian["胸软骨占比"]), 6) == 0.009234
    assert round(float(dalian["骨架占比"]), 6) == 0.580027

    tieling = rules[rules["物料号"] == "ADG0872491640"].iloc[0]
    assert tieling["来源"] == "铁岭"
    assert round(float(tieling["脖子占比"]), 6) == 0.0
    assert round(float(tieling["里肌占比"]), 6) == 0.0
    assert round(float(tieling["胸软骨占比"]), 6) == 0.008492
    assert round(float(tieling["骨架占比"]), 6) == 0.991508


def test_structure_ratio_restoration_splits_neck_tenderloin_chest_and_bone():
    ns = load_calculation_namespace()
    day = pd.Timestamp("2026-05-26")
    source_code = "ADG0872491640"
    chest_code = "AEB0630362860"
    split = pd.DataFrame(
        [
            {
                "日期": day,
                "项目": "骨架类",
                "物料号": source_code,
                "产量(kg)": 5000.0,
                "含税金额": 50000.0,
            },
            {"日期": day, "项目": "胸类-胸", "物料号": chest_code, "产量(kg)": 100.0, "含税金额": 2000.0},
            {"日期": day, "项目": "脖类", "物料号": "NECK001", "产量(kg)": 100.0, "含税金额": 800.0},
            {"日期": day, "项目": "里肌类", "物料号": "TENDER001", "产量(kg)": 50.0, "含税金额": 600.0},
        ]
    )
    m_code_daily = pd.DataFrame(
        [
            {"日期": day, "物料号": source_code, "产量(kg)": 5000.0, "综合单价_filled": 10.0},
            {"日期": day, "物料号": chest_code, "产量(kg)": 100.0, "综合单价_filled": 20.0},
        ]
    )
    rules = pd.DataFrame(
        [
            {
                "来源": "大连",
                "物料号": source_code,
                "脖子占比": 0.2,
                "里肌占比": 0.1,
                "胸软骨占比": 0.009,
                "骨架占比": 0.691,
            }
        ]
    )
    df_lw = pd.DataFrame([{"日期": day, "毛鸡净重(kg)": 100000.0}])
    chest_price = pd.DataFrame([{"日期": day, "胸软骨单价": 21.8}])

    out = ns["apply_chest_cartilage_structure_restoration"](split, m_code_daily, rules, df_lw, chest_price)

    grouped = out.groupby("项目", as_index=False)[["产量(kg)", "含税金额"]].sum()
    chest = grouped.loc[grouped["项目"] == "胸类-胸"].iloc[0]
    neck = grouped.loc[grouped["项目"] == "脖类"].iloc[0]
    tenderloin = grouped.loc[grouped["项目"] == "里肌类"].iloc[0]
    bone = grouped.loc[grouped["项目"] == "骨架类"].iloc[0]
    assert round(float(chest["产量(kg)"]), 6) == 145.0
    assert round(float(chest["含税金额"]), 6) == 2981.0
    assert round(float(neck["产量(kg)"]), 6) == 1100.0
    assert round(float(tenderloin["产量(kg)"]), 6) == 550.0
    assert round(float(bone["产量(kg)"]), 6) == 3455.0
    assert round(float(neck["含税金额"]), 6) == 8800.0
    assert round(float(tenderloin["含税金额"]), 6) == 6600.0
    assert round(float(bone["含税金额"]), 6) == 35019.0
    assert round(float(out["含税金额"].sum()), 6) == 53400.0


def test_structure_ratio_chest_cartilage_restoration_never_uses_liveweight_to_scale_deduction():
    ns = load_calculation_namespace()
    day = pd.Timestamp("2026-05-26")
    source_code = "ADG0872491640"
    chest_code = "AEB0630362860"
    split = pd.DataFrame(
        [
            {"日期": day, "项目": "骨架类", "物料号": source_code, "产量(kg)": 5000.0, "含税金额": 50000.0},
            {"日期": day, "项目": "胸类-胸", "物料号": chest_code, "产量(kg)": 1000.0, "含税金额": 20000.0},
        ]
    )
    m_code_daily = pd.DataFrame(
        [
            {"日期": day, "物料号": source_code, "产量(kg)": 5000.0, "综合单价_filled": 10.0},
            {"日期": day, "物料号": chest_code, "产量(kg)": 1000.0, "综合单价_filled": 20.0},
        ]
    )
    rules = pd.DataFrame([{"来源": "大连", "物料号": source_code, "胸软骨占比": 0.009}])
    df_lw = pd.DataFrame([{"日期": day, "毛鸡净重(kg)": 100000.0}])
    chest_price = pd.DataFrame([{"日期": day, "胸软骨单价": 21.8}])

    out = ns["apply_chest_cartilage_structure_restoration"](split, m_code_daily, rules, df_lw, chest_price)

    grouped = out.groupby("项目", as_index=False)[["产量(kg)", "含税金额"]].sum()
    chest = grouped.loc[grouped["项目"] == "胸类-胸"].iloc[0]
    bone = grouped.loc[grouped["项目"] == "骨架类"].iloc[0]
    assert round(float(chest["产量(kg)"]), 6) == 1045.0
    assert round(float(chest["含税金额"]), 6) == 20981.0
    assert round(float(bone["产量(kg)"]), 6) == 4955.0
    assert round(float(bone["含税金额"]), 6) == 49019.0


def test_chest_cartilage_manual_fallback_price_keeps_manual_value_without_109_factor():
    ns = load_calculation_namespace()
    day = pd.Timestamp("2026-05-26")
    pr_raw = pd.DataFrame(columns=["日期", "物料号", "综合单价", "金额", "数量"])
    manual = pd.DataFrame([{"物料号": "AEB0630362860", "手工单价": 20.0}])

    price = ns["build_chest_cartilage_price_by_date"](pr_raw, [day], manual)

    assert ns["tax_factor_for_code"]("AEB0630362860") == 1.09
    row = price[(price["日期"] == day) & (price["物料号"] == "AEB0630362860")].iloc[0]
    assert round(float(row["胸软骨单价"]), 6) == 20.0


def test_tieling_chest_cartilage_price_uses_tieling_code():
    ns = load_calculation_namespace()
    day = pd.Timestamp("2026-05-26")
    tieling_code = "ABB0600322058"
    pr_raw = pd.DataFrame(
        [
            {"日期": day, "物料号": tieling_code, "综合单价": 25.5, "金额": 2550.0, "数量": 100.0},
        ]
    )

    price = ns["build_chest_cartilage_price_by_date"](
        pr_raw,
        [day],
        chest_codes=[tieling_code],
    )

    row = price[(price["日期"] == day) & (price["物料号"] == tieling_code)].iloc[0]
    assert round(float(row["胸软骨单价"]), 6) == 25.5


def test_tieling_restoration_outputs_tieling_chest_cartilage_code():
    ns = load_calculation_namespace()
    day = pd.Timestamp("2026-05-26")
    source_code = "ADG0872491640"
    tieling_chest_code = "ABB0600322058"
    split = pd.DataFrame(
        [
            {"日期": day, "项目": "骨架类", "物料号": source_code, "产量(kg)": 1000.0, "含税金额": 10000.0},
        ]
    )
    m_code_daily = pd.DataFrame(
        [
            {"日期": day, "物料号": source_code, "产量(kg)": 1000.0, "综合单价_filled": 10.0},
        ]
    )
    rules = pd.DataFrame(
        [
            {
                "来源": "铁岭",
                "物料号": source_code,
                "胸软骨占比": 0.01,
                "骨架占比": 0.99,
            }
        ]
    )
    chest_price = pd.DataFrame([{"日期": day, "物料号": tieling_chest_code, "胸软骨单价": 30.0}])

    out = ns["apply_chest_cartilage_structure_restoration"](
        split,
        m_code_daily,
        rules,
        pd.DataFrame(columns=["日期", "毛鸡净重(kg)"]),
        chest_price,
    )

    chest = out[out["物料号"] == tieling_chest_code].iloc[0]
    assert chest["项目"] == "胸类-胸"
    assert round(float(chest["产量(kg)"]), 6) == 10.0
    assert round(float(chest["含税金额"]), 6) == 300.0
    assert out[out["物料号"] == "AEB0630362860"].empty


def test_primary_part_allocation_overrides_chest_cartilage_allocation_for_same_code():
    ns = load_calculation_namespace()
    fallback_alloc = pd.DataFrame(
        [
            {"日期": pd.NaT, "物料号": "ADG0872491640", "项目": "胸类-胸", "权重": 0.01},
            {"日期": pd.NaT, "物料号": "ADG0872491640", "项目": "骨架类", "权重": 0.99},
        ]
    )
    primary = pd.DataFrame(
        [
            {"日期": pd.NaT, "物料号": "ADG0872491640", "项目": "骨架类", "权重": 1.0},
        ]
    )

    merged = ns["merge_allocation_rules"](primary, fallback_alloc)
    merged_code = merged[merged["物料号"] == "ADG0872491640"]

    assert len(merged_code) == 1
    assert merged_code.iloc[0]["项目"] == "骨架类"
    assert float(merged_code.iloc[0]["权重"]) == 1.0


if __name__ == "__main__":
    test_chest_cartilage_ratio_workbook_becomes_structure_ratio_rules()
    test_structure_ratio_restoration_splits_neck_tenderloin_chest_and_bone()
    test_structure_ratio_chest_cartilage_restoration_never_uses_liveweight_to_scale_deduction()
    test_chest_cartilage_manual_fallback_price_keeps_manual_value_without_109_factor()
    test_tieling_chest_cartilage_price_uses_tieling_code()
    test_tieling_restoration_outputs_tieling_chest_cartilage_code()
    test_primary_part_allocation_overrides_chest_cartilage_allocation_for_same_code()
