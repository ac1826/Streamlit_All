import pandas as pd
import pytest

from test_chest_cartilage_month_value_conservation import load_calculation_namespace


DAY_WITH_LW = pd.Timestamp("2026-07-08")
DAY_WITHOUT_LW = pd.Timestamp("2026-07-09")


def _overview():
    rows = []
    for day, quantities in [
        (DAY_WITH_LW, {"腿类": 1000.0, "胸类-胸": 500.0, "骨架类": 200.0, "整鸡类": 0.0}),
        (DAY_WITHOUT_LW, {"腿类": -100.0, "胸类-胸": -50.0, "骨架类": 20.0, "整鸡类": 100.8}),
    ]:
        for part, qty in quantities.items():
            rows.append(
                {
                    "日期": day,
                    "项目": part,
                    "产量(kg)": qty,
                    "含税金额": qty * 10.0,
                    "含税单价": 10.0,
                }
            )
    return pd.DataFrame(rows)


def _liveweight(missing_day_value=None):
    dates = [DAY_WITH_LW]
    values = [10000.0]
    if missing_day_value is not None:
        dates.append(DAY_WITHOUT_LW)
        values.append(missing_day_value)
    return pd.DataFrame({"日期": dates, "毛鸡净重(kg)": values})


def _minors(qty=100.8, source_part="整鸡类"):
    return pd.DataFrame(
        {
            "日期": [DAY_WITHOUT_LW],
            "部位大类": [source_part],
            "子类": ["WHOLE001"],
            "产量(kg)": [qty],
        }
    )


def _mapping():
    return {"WHOLE001": ["腿类", "胸类-胸", "骨架类"]}


@pytest.mark.parametrize("missing_day_value", [None, 0.0, -10.0])
def test_missing_liveweight_whole_chicken_uses_period_structure_and_conserves_qty(missing_day_value):
    ns = load_calculation_namespace()
    overview = _overview()
    period_weights = ns["_build_restore_period_weight_dict"](
        overview, DAY_WITH_LW, DAY_WITHOUT_LW
    )

    skipped_inc, skipped_removed, _, _ = ns["_calc_restore_maps_for_day"](
        overview, _minors(), _liveweight(missing_day_value), DAY_WITHOUT_LW, _mapping()
    )
    assert skipped_inc == {}
    assert skipped_removed == {}

    inc, removed, _, _ = ns["_calc_restore_maps_for_day"](
        overview,
        _minors(),
        _liveweight(missing_day_value),
        DAY_WITHOUT_LW,
        _mapping(),
        missing_lw_whole_chicken_weights=period_weights,
    )

    assert removed["整鸡类"] == pytest.approx(100.8)
    assert sum(inc.values()) == pytest.approx(100.8)
    expected_total_weight = 900.0 + 450.0 + 220.0
    assert inc["腿类"] == pytest.approx(100.8 * 900.0 / expected_total_weight)
    assert inc["胸类-胸"] == pytest.approx(100.8 * 450.0 / expected_total_weight)
    assert inc["骨架类"] == pytest.approx(100.8 * 220.0 / expected_total_weight)


def test_missing_liveweight_negative_whole_chicken_reverses_with_same_structure():
    ns = load_calculation_namespace()
    overview = _overview()
    period_weights = ns["_build_restore_period_weight_dict"](
        overview, DAY_WITH_LW, DAY_WITHOUT_LW
    )

    inc, removed, _, _ = ns["_calc_restore_maps_for_day"](
        overview,
        _minors(qty=-100.8),
        _liveweight(),
        DAY_WITHOUT_LW,
        _mapping(),
        missing_lw_whole_chicken_weights=period_weights,
    )

    assert removed["整鸡类"] == pytest.approx(-100.8)
    assert sum(inc.values()) == pytest.approx(-100.8)
    assert all(value < 0 for value in inc.values())


def test_missing_liveweight_whole_chicken_uses_nearest_valid_day_when_period_targets_are_unavailable():
    ns = load_calculation_namespace()
    overview = _overview()
    nearest_weights = ns["_build_nearest_valid_restore_weight_dict"](
        overview,
        _liveweight(),
        DAY_WITHOUT_LW,
        DAY_WITH_LW,
        DAY_WITHOUT_LW,
    )

    inc, removed, _, _ = ns["_calc_restore_maps_for_day"](
        overview,
        _minors(),
        _liveweight(),
        DAY_WITHOUT_LW,
        _mapping(),
        missing_lw_whole_chicken_weights={"其他内脏": 1.0},
        nearest_valid_weights=nearest_weights,
    )

    assert removed["整鸡类"] == pytest.approx(100.8)
    assert sum(inc.values()) == pytest.approx(100.8)
    assert set(inc) == {"腿类", "胸类-胸", "骨架类"}


def test_missing_liveweight_fallback_does_not_restore_other_source_parts():
    ns = load_calculation_namespace()
    overview = _overview()
    period_weights = ns["_build_restore_period_weight_dict"](
        overview, DAY_WITH_LW, DAY_WITHOUT_LW
    )

    inc, removed, _, _ = ns["_calc_restore_maps_for_day"](
        overview,
        _minors(source_part="骨架类"),
        _liveweight(),
        DAY_WITHOUT_LW,
        _mapping(),
        missing_lw_whole_chicken_weights=period_weights,
    )

    assert inc == {}
    assert removed == {}


def test_period_main_side_totals_use_missing_liveweight_whole_chicken_fallback():
    ns = load_calculation_namespace()
    overview = _overview()

    without_fallback = ns["_compute_restored_main_side_qty_amt_for_period"](
        overview,
        _minors(),
        _liveweight(),
        _mapping(),
        DAY_WITH_LW,
        DAY_WITHOUT_LW,
        allow_missing_lw_whole_chicken=False,
    )
    with_fallback = ns["_compute_restored_main_side_qty_amt_for_period"](
        overview,
        _minors(),
        _liveweight(),
        _mapping(),
        DAY_WITH_LW,
        DAY_WITHOUT_LW,
        allow_missing_lw_whole_chicken=True,
    )

    main_qty_before, main_amt_before, side_qty_before, side_amt_before = without_fallback
    main_qty_after, main_amt_after, side_qty_after, side_amt_after = with_fallback
    assert main_qty_after > main_qty_before
    assert side_qty_after < side_qty_before
    assert main_qty_after + side_qty_after == pytest.approx(main_qty_before + side_qty_before)
    assert main_amt_after + side_amt_after == pytest.approx(main_amt_before + side_amt_before)
