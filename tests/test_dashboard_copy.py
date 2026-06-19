from src.reports.dashboard import position_direction_label, translate_reason


def test_up_position_is_not_described_as_short_sale():
    assert position_direction_label("UP") == "値上がり狙い（ETFを買い）"


def test_down_position_explains_inverse_etf_purchase():
    assert position_direction_label("DOWN") == "値下がり局面狙い（逆ETFを買い）"


def test_numeric_signal_reason_is_translated_to_japanese():
    assert translate_reason("numeric close-vwap and momentum candidate") == (
        "価格が当日の平均売買価格（VWAP）を上回り、"
        "短期の値動きも上向いているため"
    )
