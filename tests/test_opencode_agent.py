"""Compatibility tests for OpenCode configuration and failure behavior."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agents.external_agent import ExternalAgentAdapter, ExternalAgentNotConfiguredError
from src.config.settings import AgentConfig


def test_api_key_loading_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "env_key")

    agent = ExternalAgentAdapter(AgentConfig(type="external"), state_dir=tmp_path)

    assert agent.api_key == "env_key"


def test_missing_api_key_fails_closed(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENCODE_GO_API_KEY", raising=False)
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    with patch("src.agents.external_agent.Path") as mocked_path:
        mocked_path.return_value.exists.return_value = False
        with pytest.raises(ExternalAgentNotConfiguredError):
            ExternalAgentAdapter(AgentConfig(type="external"), state_dir=tmp_path)


@patch("src.agents.external_agent.requests.post")
def test_safe_request_converts_connection_failure_to_no_trade(
    mock_post, monkeypatch, tmp_path
):
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "key")
    agent = ExternalAgentAdapter(AgentConfig(type="external"), state_dir=tmp_path)
    agent._fetch_news = MagicMock(return_value=[{
        "id": "news-1",
        "title": "headline",
        "summary": "summary",
        "pubDate": "2026-06-18T14:00:00Z",
    }])
    mock_post.side_effect = OSError("connection refused")

    result = agent.safe_request({
        "timestamp": "2026-06-18T10:00:00-04:00",
        "symbols": {},
    })

    assert result["action"] == "NO_TRADE"
    assert result["reason"].startswith("analysis_error:")
