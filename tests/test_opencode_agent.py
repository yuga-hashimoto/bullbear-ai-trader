"""Tests for OpenCode / ExternalAgentAdapter."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.agents.external_agent import ExternalAgentAdapter, ExternalAgentNotConfiguredError
from src.config.settings import AgentConfig


def test_api_key_loading(monkeypatch):
    # Case 1: Load from env
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "env_key")
    cfg = AgentConfig(type="external")
    agent = ExternalAgentAdapter(cfg)
    assert agent.api_key == "env_key"

    # Case 2: Load from hermes env file (mocked Path)
    monkeypatch.delenv("OPENCODE_GO_API_KEY", raising=False)
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    
    with patch("src.agents.external_agent.Path") as mock_path:
        mock_file = mock_path.return_value
        mock_file.exists.return_value = True
        mock_file.open.return_value.__enter__.return_value = [
            "OPENCODE_GO_API_KEY=file_key\n",
            "OTHER_VAR=abc\n"
        ]
        agent = ExternalAgentAdapter(cfg)
        assert agent.api_key == "file_key"

    # Case 3: No key raises error (mock Path to not exist)
    with patch("src.agents.external_agent.Path") as mock_path:
        mock_file = mock_path.return_value
        mock_file.exists.return_value = False
        with pytest.raises(ExternalAgentNotConfiguredError):
            ExternalAgentAdapter(cfg)


@patch("src.agents.external_agent.requests.post")
def test_request_signal_success(mock_post, monkeypatch):
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test_key")
    cfg = AgentConfig(type="external", endpoint="http://test-api.ai")
    agent = ExternalAgentAdapter(cfg)
    agent._fetch_news = MagicMock(return_value=[{
        "id": "test-news-id",
        "title": "Test News Title",
        "summary": "Test news summary",
        "pubDate": "2026-01-01T10:00:00Z"
    }])

    # Mock successful response
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "timestamp": "2026-01-01T10:00:00-05:00",
                        "target_family": "NASDAQ",
                        "direction": "UP",
                        "action": "BUY_BULL",
                        "symbol": "TQQQ",
                        "confidence": 0.8,
                        "expected_holding_minutes": 30,
                        "reason": "Technical indicators suggest uptrend"
                    })
                }
            }
        ]
    }
    mock_post.return_value = mock_resp

    ctx = {"timestamp": "2026-01-01T10:00:00-05:00", "symbols": {}}
    res = agent.request_signal(ctx)
    assert res["action"] == "BUY_BULL"
    assert res["symbol"] == "TQQQ"
    assert res["agent_name"] == "OpenCodeAgent"


@patch("src.agents.external_agent.requests.post")
def test_request_signal_markdown_wrapping(mock_post, monkeypatch):
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test_key")
    cfg = AgentConfig(type="external")
    agent = ExternalAgentAdapter(cfg)
    agent._fetch_news = MagicMock(return_value=[{
        "id": "test-news-id",
        "title": "Test News Title",
        "summary": "Test news summary",
        "pubDate": "2026-01-01T10:00:00Z"
    }])

    # Mock response wrapped in markdown code blocks
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "```json\n" + json.dumps({
                        "timestamp": "2026-01-01T10:00:00-05:00",
                        "target_family": "MARKET",
                        "direction": "FLAT",
                        "action": "NO_TRADE",
                        "symbol": None,
                        "confidence": 0.5
                    }) + "\n```"
                }
            }
        ]
    }
    mock_post.return_value = mock_resp

    ctx = {"timestamp": "2026-01-01T10:00:00-05:00", "symbols": {}}
    res = agent.request_signal(ctx)
    assert res["action"] == "NO_TRADE"
    assert res["symbol"] is None


@patch("src.agents.external_agent.requests.post")
def test_request_signal_error_fallback(mock_post, monkeypatch):
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test_key")
    cfg = AgentConfig(type="external")
    agent = ExternalAgentAdapter(cfg)
    agent._fetch_news = MagicMock(return_value=[{
        "id": "test-news-id",
        "title": "Test News Title",
        "summary": "Test news summary",
        "pubDate": "2026-01-01T10:00:00Z"
    }])

    # Mock connection failure
    mock_post.side_effect = Exception("Connection refused")

    ctx = {"timestamp": "2026-01-01T10:00:00-05:00", "symbols": {}}
    
    # request_signal should raise
    with pytest.raises(RuntimeError):
        agent.request_signal(ctx)

    # safe_request should catch it and fallback to NO_TRADE
    fallback_res = agent.safe_request(ctx)
    assert fallback_res["action"] == "NO_TRADE"
    assert fallback_res["symbol"] is None
    assert "fallback:" in fallback_res["reason"]


@patch("src.agents.external_agent.requests.post")
def test_news_triggered_skipping(mock_post, monkeypatch):
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test_key")
    cfg = AgentConfig(type="external")
    agent = ExternalAgentAdapter(cfg)

    # 1. ニュースがない場合の検証 -> APIを呼ばずに即座にフォールバック（MockAgentが方向性エッジ無しで NO_TRADE を返す）
    agent._fetch_news = MagicMock(return_value=[])
    ctx = {"timestamp": "2026-01-01T10:05:00-05:00", "symbols": {}}
    res_no_news = agent.request_signal(ctx)
    
    assert res_no_news["action"] == "NO_TRADE"
    assert "no directional edge" in res_no_news["reason"]
    mock_post.assert_not_called()

    # 1.1 ニュースがなく、かつMockAgentで上昇条件を満たす場合の検証 -> MockAgentがBUY_BULL (TQQQ) を返す
    ctx_bull = {
        "timestamp": "2026-01-01T10:05:00-05:00",
        "symbols": {
            "QQQ": {
                "close": 400.0,
                "vwap": 395.0,
                "returns": {"3_bar": 0.01}
            }
        }
    }
    res_bull = agent.request_signal(ctx_bull)
    assert res_bull["action"] == "BUY_BULL"
    assert res_bull["symbol"] == "TQQQ"
    assert res_bull["agent_name"] == "MockAgent"
    mock_post.assert_not_called()

    # 2. ニュースがある場合の検証 -> APIを呼ぶ
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {"message": {"content": json.dumps({
                "timestamp": "2026-01-01T10:00:00-05:00",
                "target_family": "NASDAQ",
                "direction": "UP",
                "action": "BUY_BULL",
                "symbol": "TQQQ",
                "confidence": 0.8
            })}}
        ]
    }
    mock_post.return_value = mock_resp
    
    agent._fetch_news = MagicMock(return_value=[{
        "id": "news-123",
        "title": "Important News",
        "summary": "Big move expected",
        "pubDate": "2026-01-01T10:00:00Z"
    }])
    
    res_with_news = agent.request_signal(ctx)
    assert res_with_news["action"] == "BUY_BULL"
    assert mock_post.call_count == 1

    # 3. 重複排除の検証 -> 同じニュースIDがあるが、すでに seen_news_ids に登録されているため
    # 次回呼び出し時は API を呼ばずに MockAgent へのフォールバックが発生する
    mock_post.reset_mock()
    res_duplicate = agent.request_signal(ctx)
    assert res_duplicate["action"] == "NO_TRADE"
    assert "no directional edge" in res_duplicate["reason"]
    mock_post.assert_not_called()

