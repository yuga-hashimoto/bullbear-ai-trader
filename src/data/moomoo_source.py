"""moomoo (Futu) OpenAPI OHLCV adapter — skeleton.

The moomoo OpenAPI is accessed through a locally running OpenD gateway via the
``futu`` Python SDK. This adapter is intentionally a stub: it documents the
intended integration but raises ``NotImplementedError`` until wired up, so it
can never silently fetch/trade. Historical-data retrieval here is READ-ONLY and
unrelated to order placement, but we keep it disabled until explicitly built.
"""
from __future__ import annotations

import pandas as pd

from .base import DataSource

_INTERVAL_TO_KTYPE = {
    "1m": "K_1M",
    "5m": "K_5M",
    "15m": "K_15M",
}


class MoomooDataSource(DataSource):
    """Future moomoo OpenD historical-data adapter (not yet implemented)."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 11111,
        tz: str = "America/New_York",
    ) -> None:
        self.host = host
        self.port = port
        self.tz = tz

    def fetch(self, symbol: str, interval: str, start: str, end: str) -> pd.DataFrame:
        # Intended implementation outline (kept as guidance, not active code):
        #   from futu import OpenQuoteContext, SubType, KLType
        #   ctx = OpenQuoteContext(host=self.host, port=self.port)
        #   ret, df, _ = ctx.request_history_kline(
        #       code=f"US.{symbol}", start=start, end=end,
        #       ktype=KLType.[_INTERVAL_TO_KTYPE[interval]], ...)
        #   ctx.close()
        #   -> normalize to OHLCV_COLUMNS, localize to self.tz
        raise NotImplementedError(
            "MoomooDataSource is a future-use skeleton. Use data_source: "
            "yfinance or synthetic for now. Supported ktype mapping: "
            f"{_INTERVAL_TO_KTYPE}."
        )
