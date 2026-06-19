from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.agents.analysis_schema import MarketAnalysis, MarketAnalysisError
from src.agents.external_agent import ExternalAgentAdapter
from src.config.settings import AgentConfig


def test_no_news_returns_neutral_no_trade(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "key")
    agent = ExternalAgentAdapter(AgentConfig(type="external"), state_dir=tmp_path)
    agent._fetch_news = MagicMock(return_value=[])

    result = agent.request_signal({"timestamp": "2026-06-18T10:00:00-04:00", "symbols": {}})

    assert result["action"] == "NO_TRADE"
    assert result["reason"] == "no_new_news"


def test_seen_news_ids_survive_restart(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "key")
    cfg = AgentConfig(type="external")
    first = ExternalAgentAdapter(cfg, state_dir=tmp_path)
    first.news_store.mark_seen(["news-1"])

    second = ExternalAgentAdapter(cfg, state_dir=tmp_path)

    assert second.news_store.is_seen("news-1")


def test_expired_analysis_is_not_actionable():
    analysis = MarketAnalysis.from_dict({
        "timestamp": "2026-06-18T10:00:00-04:00",
        "valid_until": "2026-06-18T10:05:00-04:00",
        "target_family": "NASDAQ",
        "direction": "UP",
        "confidence": 0.8,
        "thesis": "positive catalyst",
        "invalidation": "price rejects catalyst",
        "risk_factors": [],
        "source_news_ids": ["news-1"],
    })

    assert not analysis.is_valid_at(pd.Timestamp("2026-06-18T10:06:00-04:00"))


def test_analysis_schema_rejects_order_fields():
    with pytest.raises(MarketAnalysisError):
        MarketAnalysis.from_dict({
            "timestamp": "2026-06-18T10:00:00-04:00",
            "valid_until": "2026-06-18T10:05:00-04:00",
            "target_family": "NASDAQ",
            "direction": "UP",
            "confidence": 0.8,
            "thesis": "x",
            "invalidation": "y",
            "risk_factors": [],
            "source_news_ids": ["n"],
            "action": "BUY_BULL",
        })


@patch("src.agents.external_agent.requests.post")
def test_opencode_analysis_is_embedded_in_no_trade_signal(mock_post, monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "key")
    agent = ExternalAgentAdapter(
        AgentConfig(type="external", endpoint="https://example.invalid"),
        state_dir=tmp_path,
    )
    agent._fetch_news = MagicMock(return_value=[{
        "id": "news-1",
        "title": "Catalyst",
        "summary": "Positive",
        "pubDate": "2026-06-18T14:00:00Z",
    }])
    response = MagicMock()
    response.json.return_value = {"choices": [{"message": {"content": json.dumps({
        "timestamp": "2026-06-18T10:00:00-04:00",
        "valid_until": "2026-06-18T10:30:00-04:00",
        "target_family": "NASDAQ",
        "direction": "UP",
        "confidence": 0.8,
        "thesis": "positive catalyst",
        "invalidation": "market rejects news",
        "risk_factors": ["headline reversal"],
        "source_news_ids": ["news-1"],
    })}}]}
    mock_post.return_value = response

    result = agent.request_signal({
        "timestamp": "2026-06-18T10:00:00-04:00",
        "symbols": {},
    })

    assert result["action"] == "NO_TRADE"
    assert result["raw_response"]["analysis"]["direction"] == "UP"
    assert agent.news_store.is_seen("news-1")


@patch("src.agents.external_agent.requests.post")
def test_opencode_accepts_relevant_subset_of_supplied_news_ids(
    mock_post, monkeypatch, tmp_path
):
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "key")
    agent = ExternalAgentAdapter(
        AgentConfig(type="external", endpoint="https://example.invalid"),
        state_dir=tmp_path,
    )
    agent._fetch_news = MagicMock(return_value=[
        {"id": "news-1", "title": "Relevant", "summary": "", "pubDate": ""},
        {"id": "news-2", "title": "Irrelevant", "summary": "", "pubDate": ""},
    ])
    response = MagicMock()
    response.json.return_value = {"choices": [{"message": {"content": json.dumps({
        "timestamp": "2026-06-18T10:00:00-04:00",
        "valid_until": "2026-06-18T10:30:00-04:00",
        "target_family": "NASDAQ",
        "direction": "UP",
        "confidence": 0.8,
        "thesis": "news-1 is the relevant catalyst",
        "invalidation": "market rejects news",
        "risk_factors": [],
        "source_news_ids": ["news-1"],
    })}}]}
    mock_post.return_value = response

    result = agent.request_signal({
        "timestamp": "2026-06-18T10:00:00-04:00",
        "symbols": {},
    })

    assert result["raw_response"]["analysis"]["source_news_ids"] == ["news-1"]
    assert agent.news_store.is_seen("news-1")
    assert agent.news_store.is_seen("news-2")


@patch("src.agents.external_agent.requests.post")
def test_opencode_rejects_unknown_source_news_id(
    mock_post, monkeypatch, tmp_path
):
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "key")
    agent = ExternalAgentAdapter(
        AgentConfig(type="external", endpoint="https://example.invalid"),
        state_dir=tmp_path,
    )
    agent._fetch_news = MagicMock(return_value=[
        {"id": "news-1", "title": "Known", "summary": "", "pubDate": ""},
    ])
    response = MagicMock()
    response.json.return_value = {"choices": [{"message": {"content": json.dumps({
        "timestamp": "2026-06-18T10:00:00-04:00",
        "valid_until": "2026-06-18T10:30:00-04:00",
        "target_family": "NASDAQ",
        "direction": "UP",
        "confidence": 0.8,
        "thesis": "fabricated citation",
        "invalidation": "market rejects news",
        "risk_factors": [],
        "source_news_ids": ["unknown-news"],
    })}}]}
    mock_post.return_value = response

    with pytest.raises(ValueError, match="unknown source_news_ids"):
        agent.request_signal({
            "timestamp": "2026-06-18T10:00:00-04:00",
            "symbols": {},
        })
