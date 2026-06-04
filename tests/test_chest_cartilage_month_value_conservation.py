from pathlib import Path
import sys
import types

import pandas as pd


APP_PATH = Path(r"F:/llqdocument/new modle/app_streamlit.py")


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


def test_month_code_detail_keeps_source_value_after_chest_restoration():
    ns = load_calculation_namespace()
    source_code = "ADG_SOURCE"
    chest_code = "AEB0630362860"
    minor_month = pd.DataFrame(
        [
            {"项目": "骨架类", "物料号": source_code, "产量(kg)": 900.0},
            {"项目": "胸类-胸", "物料号": chest_code, "产量(kg)": 150.0},
        ]
    )
    price_code_mtd = pd.DataFrame(
        [
            {"物料号": source_code, "月均价": 4.0},
            {"物料号": chest_code, "月均价": 20.0},
        ]
    )
    rules = pd.DataFrame(
        [
            {
                "来源": "大连",
                "物料号": source_code,
                "胸软骨占比": 0.1,
                "骨架占比": 0.9,
                "脖子占比": 0.0,
                "里肌占比": 0.0,
            }
        ]
    )

    detail = ns["build_month_code_detail_with_restoration_value_conservation"](
        minor_month,
        price_code_mtd,
        rules,
        part_col="项目",
    )

    total_amount = float(detail["含税金额"].sum())
    expected_source_amount = 1000.0 * 4.0
    expected_existing_chest_amount = 50.0 * 20.0
    assert round(total_amount, 6) == round(expected_source_amount + expected_existing_chest_amount, 6)

    bone = detail[(detail["项目"] == "骨架类") & (detail["物料号"] == source_code)].iloc[0]
    assert round(float(bone["产量(kg)"]), 6) == 900.0
    assert round(float(bone["含税金额"]), 6) == 2000.0
    assert round(float(bone["月均价"]), 6) == round(2000.0 / 900.0, 6)


def test_month_code_detail_keeps_ordinary_items_on_month_price():
    ns = load_calculation_namespace()
    minor_month = pd.DataFrame(
        [
            {"项目": "腿类", "物料号": "LEG001", "产量(kg)": 100.0},
            {"项目": "腿类", "物料号": "LEG001", "产量(kg)": 50.0},
        ]
    )
    price_code_mtd = pd.DataFrame([{"物料号": "LEG001", "月均价": 8.0}])

    detail = ns["build_month_code_detail_with_restoration_value_conservation"](
        minor_month,
        price_code_mtd,
        pd.DataFrame(),
        part_col="项目",
    )

    row = detail.iloc[0]
    assert round(float(row["产量(kg)"]), 6) == 150.0
    assert round(float(row["含税金额"]), 6) == 1200.0
    assert round(float(row["月均价"]), 6) == 8.0


def test_month_code_detail_records_negative_source_remainder_warning():
    ns = load_calculation_namespace()
    source_code = "ADG_SOURCE"
    chest_code = "AEB0630362860"
    minor_month = pd.DataFrame(
        [
            {"项目": "骨架类", "物料号": source_code, "产量(kg)": 500.0},
            {"项目": "胸类-胸", "物料号": chest_code, "产量(kg)": 500.0},
        ]
    )
    price_code_mtd = pd.DataFrame(
        [
            {"物料号": source_code, "月均价": 2.0},
            {"物料号": chest_code, "月均价": 50.0},
        ]
    )
    rules = pd.DataFrame(
        [
            {
                "来源": "大连",
                "物料号": source_code,
                "胸软骨占比": 0.5,
                "骨架占比": 0.5,
                "脖子占比": 0.0,
                "里肌占比": 0.0,
            }
        ]
    )

    detail = ns["build_month_code_detail_with_restoration_value_conservation"](
        minor_month,
        price_code_mtd,
        rules,
        part_col="项目",
    )

    warnings = detail.attrs["chest_restoration_negative_value_warnings"]
    assert len(warnings) == 1
    assert warnings[0]["物料号"] == source_code
    assert warnings[0]["调整后金额"] < 0


def test_month_conservation_adjusts_only_rows_marked_as_restored():
    ns = load_calculation_namespace()
    source_code = "ADG_SOURCE"
    chest_code = "AEB0630362860"
    minor_month = pd.DataFrame(
        [
            {
                "项目": "骨架类",
                "物料号": source_code,
                "产量(kg)": 900.0,
                "胸软骨还原来源物料号": source_code,
                "胸软骨还原胸软骨物料号": chest_code,
                "胸软骨还原胸软骨占比": 0.1,
                "胸软骨还原非胸占比": 0.9,
            },
            {
                "项目": "胸类-胸",
                "物料号": chest_code,
                "产量(kg)": 100.0,
                "胸软骨还原来源物料号": source_code,
                "胸软骨还原胸软骨物料号": chest_code,
                "胸软骨还原胸软骨占比": 0.1,
                "胸软骨还原非胸占比": 0.9,
            },
            {"项目": "骨架类", "物料号": source_code, "产量(kg)": 1000.0},
        ]
    )
    price_code_mtd = pd.DataFrame(
        [
            {"物料号": source_code, "月均价": 4.0},
            {"物料号": chest_code, "月均价": 20.0},
        ]
    )
    rules = pd.DataFrame(
        [
            {
                "来源": "大连",
                "物料号": source_code,
                "胸软骨占比": 0.1,
                "骨架占比": 0.9,
            }
        ]
    )

    detail = ns["build_month_code_detail_with_restoration_value_conservation"](
        minor_month,
        price_code_mtd,
        rules,
        part_col="项目",
    )

    total_amount = float(detail["含税金额"].sum())
    assert round(total_amount, 6) == 8000.0


def test_restored_month_rows_use_matching_part_month_price_and_keep_source_total():
    ns = load_calculation_namespace()
    source_code = "ADG_SOURCE"
    chest_code = "AEB0630362860"
    minor_month = pd.DataFrame(
        [
            {"项目": "脖类", "物料号": "NECK_BASE", "产量(kg)": 100.0},
            {"项目": "里肌类", "物料号": "TENDER_BASE", "产量(kg)": 100.0},
            {"项目": "胸类-胸", "物料号": "CHEST_BASE", "产量(kg)": 100.0},
            {
                "项目": "脖类",
                "物料号": source_code,
                "产量(kg)": 200.0,
                "胸软骨还原来源物料号": source_code,
                "胸软骨还原胸软骨物料号": chest_code,
                "胸软骨还原胸软骨占比": 0.1,
                "胸软骨还原非胸占比": 0.9,
            },
            {
                "项目": "里肌类",
                "物料号": source_code,
                "产量(kg)": 300.0,
                "胸软骨还原来源物料号": source_code,
                "胸软骨还原胸软骨物料号": chest_code,
                "胸软骨还原胸软骨占比": 0.1,
                "胸软骨还原非胸占比": 0.9,
            },
            {
                "项目": "骨架类",
                "物料号": source_code,
                "产量(kg)": 400.0,
                "胸软骨还原来源物料号": source_code,
                "胸软骨还原胸软骨物料号": chest_code,
                "胸软骨还原胸软骨占比": 0.1,
                "胸软骨还原非胸占比": 0.9,
            },
            {
                "项目": "胸类-胸",
                "物料号": chest_code,
                "产量(kg)": 100.0,
                "胸软骨还原来源物料号": source_code,
                "胸软骨还原胸软骨物料号": chest_code,
                "胸软骨还原胸软骨占比": 0.1,
                "胸软骨还原非胸占比": 0.9,
            },
        ]
    )
    price_code_mtd = pd.DataFrame(
        [
            {"物料号": source_code, "月均价": 4.0},
            {"物料号": chest_code, "月均价": 20.0},
            {"物料号": "NECK_BASE", "月均价": 6.0},
            {"物料号": "TENDER_BASE", "月均价": 12.0},
            {"物料号": "CHEST_BASE", "月均价": 30.0},
        ]
    )
    rules = pd.DataFrame(
        [
            {
                "来源": "大连",
                "物料号": source_code,
                "胸软骨占比": 0.1,
                "脖子占比": 0.2,
                "里肌占比": 0.3,
                "骨架占比": 0.4,
            }
        ]
    )

    detail = ns["build_month_code_detail_with_restoration_value_conservation"](
        minor_month,
        price_code_mtd,
        rules,
        part_col="项目",
    )

    neck = detail[(detail["项目"] == "脖类") & (detail["物料号"] == source_code)].iloc[0]
    tender = detail[(detail["项目"] == "里肌类") & (detail["物料号"] == source_code)].iloc[0]
    chest = detail[(detail["项目"] == "胸类-胸") & (detail["物料号"] == chest_code)].iloc[0]
    bone = detail[(detail["项目"] == "骨架类") & (detail["物料号"] == source_code)].iloc[0]

    assert round(float(neck["月均价"]), 6) == 6.0
    assert round(float(tender["月均价"]), 6) == 12.0
    assert round(float(chest["月均价"]), 6) == 20.0
    assert round(float(bone["含税金额"]), 6) == -2800.0
    source_group_amount = (
        float(neck["含税金额"])
        + float(tender["含税金额"])
        + float(chest["含税金额"])
        + float(bone["含税金额"])
    )
    assert round(source_group_amount, 6) == 4000.0


def test_restored_month_rows_record_warning_when_part_price_falls_back():
    ns = load_calculation_namespace()
    source_code = "ADG_SOURCE"
    chest_code = "AEB0630362860"
    minor_month = pd.DataFrame(
        [
            {
                "项目": "里肌类",
                "物料号": source_code,
                "产量(kg)": 300.0,
                "胸软骨还原来源物料号": source_code,
                "胸软骨还原胸软骨物料号": chest_code,
                "胸软骨还原胸软骨占比": 0.1,
                "胸软骨还原非胸占比": 0.9,
            },
            {
                "项目": "骨架类",
                "物料号": source_code,
                "产量(kg)": 600.0,
                "胸软骨还原来源物料号": source_code,
                "胸软骨还原胸软骨物料号": chest_code,
                "胸软骨还原胸软骨占比": 0.1,
                "胸软骨还原非胸占比": 0.9,
            },
            {
                "项目": "胸类-胸",
                "物料号": chest_code,
                "产量(kg)": 100.0,
                "胸软骨还原来源物料号": source_code,
                "胸软骨还原胸软骨物料号": chest_code,
                "胸软骨还原胸软骨占比": 0.1,
                "胸软骨还原非胸占比": 0.9,
            },
        ]
    )
    price_code_mtd = pd.DataFrame(
        [
            {"物料号": source_code, "月均价": 4.0},
            {"物料号": chest_code, "月均价": 20.0},
        ]
    )
    rules = pd.DataFrame(
        [
            {
                "来源": "大连",
                "物料号": source_code,
                "胸软骨占比": 0.1,
                "里肌占比": 0.3,
                "骨架占比": 0.6,
            }
        ]
    )

    detail = ns["build_month_code_detail_with_restoration_value_conservation"](
        minor_month,
        price_code_mtd,
        rules,
        part_col="项目",
    )

    tender = detail[(detail["项目"] == "里肌类") & (detail["物料号"] == source_code)].iloc[0]
    assert round(float(tender["月均价"]), 6) == 4.0
    warnings = detail.attrs["chest_restoration_missing_part_price_warnings"]
    assert warnings
    assert warnings[0]["部位"] == "里肌类"


def test_restored_month_chest_uses_chest_code_price_not_chest_part_average():
    ns = load_calculation_namespace()
    source_code = "ADG_SOURCE"
    chest_code = "AEB0630362860"
    minor_month = pd.DataFrame(
        [
            {"项目": "胸类-胸", "物料号": "CHEST_BASE", "产量(kg)": 100.0},
            {
                "项目": "骨架类",
                "物料号": source_code,
                "产量(kg)": 900.0,
                "胸软骨还原来源物料号": source_code,
                "胸软骨还原胸软骨物料号": chest_code,
                "胸软骨还原胸软骨占比": 0.1,
                "胸软骨还原非胸占比": 0.9,
            },
            {
                "项目": "胸类-胸",
                "物料号": chest_code,
                "产量(kg)": 100.0,
                "胸软骨还原来源物料号": source_code,
                "胸软骨还原胸软骨物料号": chest_code,
                "胸软骨还原胸软骨占比": 0.1,
                "胸软骨还原非胸占比": 0.9,
            },
        ]
    )
    price_code_mtd = pd.DataFrame(
        [
            {"物料号": source_code, "月均价": 4.0},
            {"物料号": chest_code, "月均价": 21.8},
            {"物料号": "CHEST_BASE", "月均价": 30.0},
        ]
    )
    rules = pd.DataFrame(
        [
            {
                "来源": "大连",
                "物料号": source_code,
                "胸软骨占比": 0.1,
                "骨架占比": 0.9,
            }
        ]
    )

    detail = ns["build_month_code_detail_with_restoration_value_conservation"](
        minor_month,
        price_code_mtd,
        rules,
        part_col="项目",
    )

    chest = detail[(detail["项目"] == "胸类-胸") & (detail["物料号"] == chest_code)].iloc[0]
    bone = detail[(detail["项目"] == "骨架类") & (detail["物料号"] == source_code)].iloc[0]
    assert round(float(chest["月均价"]), 6) == 21.8
    assert round(float(chest["含税金额"]), 6) == 2180.0
    assert round(float(bone["含税金额"]) + float(chest["含税金额"]), 6) == 4000.0


def test_tieling_adg0875491660_month_remainder_keeps_source_value():
    ns = load_calculation_namespace()
    source_code = "ADG0875491660"
    chest_code = "ABB0600322058"
    source_original_qty = 3883.0
    source_original_amount = 22683.768363
    restored_chest_qty = 32.975603
    restored_chest_amount = 841.044849
    restored_bone_qty = 3850.024397
    expected_bone_amount = source_original_amount - restored_chest_amount

    minor_month = pd.DataFrame(
        [
            {
                "项目": "骨架类",
                "物料号": source_code,
                "产量(kg)": restored_bone_qty,
                "胸软骨还原来源物料号": source_code,
                "胸软骨还原胸软骨物料号": chest_code,
                "胸软骨还原胸软骨占比": restored_chest_qty / source_original_qty,
                "胸软骨还原非胸占比": restored_bone_qty / source_original_qty,
            },
            {
                "项目": "胸类-胸",
                "物料号": chest_code,
                "产量(kg)": restored_chest_qty,
                "胸软骨还原来源物料号": source_code,
                "胸软骨还原胸软骨物料号": chest_code,
                "胸软骨还原胸软骨占比": restored_chest_qty / source_original_qty,
                "胸软骨还原非胸占比": restored_bone_qty / source_original_qty,
            },
        ]
    )
    price_code_mtd = pd.DataFrame(
        [
            {"物料号": source_code, "月均价": source_original_amount / source_original_qty},
            {"物料号": chest_code, "月均价": restored_chest_amount / restored_chest_qty},
        ]
    )
    rules = pd.DataFrame(
        [
            {
                "来源": "铁岭",
                "物料号": source_code,
                "胸软骨占比": restored_chest_qty / source_original_qty,
                "骨架占比": restored_bone_qty / source_original_qty,
            }
        ]
    )

    detail = ns["build_month_code_detail_with_restoration_value_conservation"](
        minor_month,
        price_code_mtd,
        rules,
        part_col="项目",
    )

    chest = detail[(detail["项目"] == "胸类-胸") & (detail["物料号"] == chest_code)].iloc[0]
    bone = detail[(detail["项目"] == "骨架类") & (detail["物料号"] == source_code)].iloc[0]
    assert round(float(chest["含税金额"]), 6) == round(restored_chest_amount, 6)
    assert round(float(bone["含税金额"]), 6) == round(expected_bone_amount, 6)
    assert round(float(chest["含税金额"]) + float(bone["含税金额"]), 6) == round(source_original_amount, 6)
    adjustments = detail.attrs["chest_restoration_conservation_adjustments"]
    assert len(adjustments) == 1
    assert adjustments[0]["物料号"] == source_code
    assert round(float(adjustments[0]["来源原金额"]), 6) == round(source_original_amount, 6)
    assert round(float(adjustments[0]["骨架承接金额"]), 6) == round(expected_bone_amount, 6)
    assert round(float(detail.attrs["chest_restoration_total_target_amount"]), 6) == round(source_original_amount, 6)


def test_month_detail_target_amount_includes_ordinary_rows_and_restored_source_value():
    ns = load_calculation_namespace()
    source_code = "ADG_SOURCE"
    chest_code = "AEB0630362860"
    minor_month = pd.DataFrame(
        [
            {"项目": "腿类", "物料号": "LEG001", "产量(kg)": 100.0},
            {
                "项目": "骨架类",
                "物料号": source_code,
                "产量(kg)": 900.0,
                "胸软骨还原来源物料号": source_code,
                "胸软骨还原胸软骨物料号": chest_code,
                "胸软骨还原胸软骨占比": 0.1,
                "胸软骨还原非胸占比": 0.9,
            },
            {
                "项目": "胸类-胸",
                "物料号": chest_code,
                "产量(kg)": 100.0,
                "胸软骨还原来源物料号": source_code,
                "胸软骨还原胸软骨物料号": chest_code,
                "胸软骨还原胸软骨占比": 0.1,
                "胸软骨还原非胸占比": 0.9,
            },
        ]
    )
    price_code_mtd = pd.DataFrame(
        [
            {"物料号": "LEG001", "月均价": 8.0},
            {"物料号": source_code, "月均价": 4.0},
            {"物料号": chest_code, "月均价": 20.0},
        ]
    )
    rules = pd.DataFrame(
        [
            {
                "来源": "大连",
                "物料号": source_code,
                "胸软骨占比": 0.1,
                "骨架占比": 0.9,
            }
        ]
    )

    detail = ns["build_month_code_detail_with_restoration_value_conservation"](
        minor_month,
        price_code_mtd,
        rules,
        part_col="项目",
    )

    expected_total = 100.0 * 8.0 + 1000.0 * 4.0
    assert round(float(detail.attrs["chest_restoration_total_target_amount"]), 6) == round(expected_total, 6)
    assert round(float(detail["含税金额"].sum()), 6) == round(expected_total, 6)


def test_reconcile_month_project_amount_total_pushes_delta_to_bone_and_combos():
    ns = load_calculation_namespace()
    current = pd.DataFrame(
        [
            {"项目": "胸类-胸", "产量(kg)": 100.0, "含税金额": 1000.0},
            {"项目": "骨架类", "产量(kg)": 900.0, "含税金额": 1500.0},
            {"项目": "鸡头类", "产量(kg)": 10.0, "含税金额": 20.0},
            {"项目": "脖类", "产量(kg)": 20.0, "含税金额": 30.0},
            {"项目": "胸类", "产量(kg)": 100.0, "含税金额": 1000.0},
            {"项目": "鸡头+鸡脖+骨架", "产量(kg)": 930.0, "含税金额": 1550.0},
        ]
    )

    out = ns["reconcile_month_project_amount_total"](current, 4100.0)

    bone = out[out["项目"] == "骨架类"].iloc[0]
    combo = out[out["项目"] == "鸡头+鸡脖+骨架"].iloc[0]
    base = out[out["项目"].isin(["胸类", "骨架类", "鸡头类", "脖类"])]
    assert round(float(bone["含税金额"]), 6) == 3050.0
    assert round(float(combo["含税金额"]), 6) == 3100.0
    assert round(float(base["含税金额"].sum()), 6) == 4100.0


def test_restored_month_negative_reversal_rows_reduce_chest_and_bone_amounts():
    ns = load_calculation_namespace()
    source_code = "ADG_SOURCE"
    chest_code = "ABB0600322058"
    minor_month = pd.DataFrame(
        [
            {
                "项目": "骨架类",
                "物料号": source_code,
                "产量(kg)": 900.0,
                "胸软骨还原来源物料号": source_code,
                "胸软骨还原胸软骨物料号": chest_code,
                "胸软骨还原胸软骨占比": 0.1,
                "胸软骨还原非胸占比": 0.9,
            },
            {
                "项目": "胸类-胸",
                "物料号": chest_code,
                "产量(kg)": 100.0,
                "胸软骨还原来源物料号": source_code,
                "胸软骨还原胸软骨物料号": chest_code,
                "胸软骨还原胸软骨占比": 0.1,
                "胸软骨还原非胸占比": 0.9,
            },
            {
                "项目": "骨架类",
                "物料号": source_code,
                "产量(kg)": -90.0,
                "胸软骨还原来源物料号": source_code,
                "胸软骨还原胸软骨物料号": chest_code,
                "胸软骨还原胸软骨占比": 0.1,
                "胸软骨还原非胸占比": 0.9,
            },
            {
                "项目": "胸类-胸",
                "物料号": chest_code,
                "产量(kg)": -10.0,
                "胸软骨还原来源物料号": source_code,
                "胸软骨还原胸软骨物料号": chest_code,
                "胸软骨还原胸软骨占比": 0.1,
                "胸软骨还原非胸占比": 0.9,
            },
        ]
    )
    price_code_mtd = pd.DataFrame(
        [
            {"物料号": source_code, "月均价": 4.0},
            {"物料号": chest_code, "月均价": 20.0},
        ]
    )
    rules = pd.DataFrame(
        [
            {
                "来源": "铁岭",
                "物料号": source_code,
                "胸软骨占比": 0.1,
                "骨架占比": 0.9,
            }
        ]
    )

    detail = ns["build_month_code_detail_with_restoration_value_conservation"](
        minor_month,
        price_code_mtd,
        rules,
        part_col="项目",
    )

    chest = detail[(detail["项目"] == "胸类-胸") & (detail["物料号"] == chest_code)].iloc[0]
    bone = detail[(detail["项目"] == "骨架类") & (detail["物料号"] == source_code)].iloc[0]
    assert round(float(chest["含税金额"]), 6) == 1800.0
    assert round(float(bone["含税金额"]), 6) == 1800.0
    assert round(float(chest["含税金额"]) + float(bone["含税金额"]), 6) == 3600.0


if __name__ == "__main__":
    test_month_code_detail_keeps_source_value_after_chest_restoration()
    test_month_code_detail_keeps_ordinary_items_on_month_price()
    test_month_code_detail_records_negative_source_remainder_warning()
    test_month_conservation_adjusts_only_rows_marked_as_restored()
    test_restored_month_rows_use_matching_part_month_price_and_keep_source_total()
    test_restored_month_rows_record_warning_when_part_price_falls_back()
    test_restored_month_chest_uses_chest_code_price_not_chest_part_average()
    test_tieling_adg0875491660_month_remainder_keeps_source_value()
    test_month_detail_target_amount_includes_ordinary_rows_and_restored_source_value()
    test_reconcile_month_project_amount_total_pushes_delta_to_bone_and_combos()
    test_restored_month_negative_reversal_rows_reduce_chest_and_bone_amounts()
