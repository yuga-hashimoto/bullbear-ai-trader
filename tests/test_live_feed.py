from __future__ import annotations

import dataclasses

from src.runners.feed import SyntheticLiveFeed, YFinanceLiveFeed, make_live_feed


def test_feed_factory_uses_real_feed_for_yfinance_config(cfg):
    real_cfg = dataclasses.replace(cfg, data_source="yfinance")

    assert isinstance(make_live_feed(real_cfg), YFinanceLiveFeed)
    assert isinstance(make_live_feed(cfg), SyntheticLiveFeed)
