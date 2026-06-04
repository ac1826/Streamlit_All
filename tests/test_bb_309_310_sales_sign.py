from pathlib import Path
import sys
import types

import pandas as pd


APP_PATH = Path(r"F:/llqdocument/new modle/app_streamlit.py")
SAMPLE_PATH = Path(r"F:/llqdocument/大成文件/SAP调取/BB/2026-05-18_销量_BB_309.xlsx")


def load_calculation_namespace():
    source = APP_PATH.read_text(encoding="utf-8-sig")
    cutoff = source.index('st.sidebar.header("数据源（请上传）")')
    module = types.ModuleType("app_streamlit_calculation")
    streamlit_stub = types.SimpleNamespace(
        sidebar=types.SimpleNamespace(file_uploader=lambda *args, **kwargs: None),
        cache_data=lambda *args, **kwargs: (lambda fn: fn),
        cache_resource=lambda *args, **kwargs: (lambda fn: fn),
    )
    sys.modules.setdefault("streamlit", streamlit_stub)
    exec(compile(source[:cutoff], str(APP_PATH), "exec"), module.__dict__)
    return module.__dict__


def test_bb_309_file_adds_mvt_309_and_subtracts_mvt_310_for_a_materials():
    ns = load_calculation_namespace()
    df = pd.read_excel(SAMPLE_PATH)
    df["__source_file__"] = SAMPLE_PATH.name

    actual = ns["_aggregate_sales_rows"](df)

    assert round(actual["销售净重"].sum(), 6) == 409955.0
    assert round(actual["销售金额"].sum(), 4) == 3824579.1707

    day_0516 = actual.loc[actual["日期"] == pd.Timestamp("2026-05-16")]
    assert round(day_0516["销售净重"].sum(), 6) == 28550.0
    assert round(day_0516["销售金额"].sum(), 4) == 282627.9857


if __name__ == "__main__":
    test_bb_309_file_adds_mvt_309_and_subtracts_mvt_310_for_a_materials()
