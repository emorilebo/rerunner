"""Unit tests for automated secret, API key, and PII redactor."""

from __future__ import annotations

import re
from rerunner.sanitizer import (
    sanitize_chunks,
    sanitize_headers,
    sanitize_payload,
    sanitize_string,
    sanitize_url,
)


def test_sanitize_anthropic_and_openai_keys() -> None:
    ant_key = "sk-ant-" + "api03-abcdef1234567890abcdef1234567890"
    proj_key = "sk-proj-" + "1234567890abcdef1234567890123456"
    text = f"Using key {ant_key} and {proj_key}"
    cleaned = sanitize_string(text)
    assert "sk-ant" not in cleaned
    assert "[REDACTED_ANTHROPIC_KEY]" in cleaned
    assert "sk-proj" not in cleaned
    assert "[REDACTED_OPENAI_PROJECT_KEY]" in cleaned


def test_sanitize_google_slack_and_cloud_tokens() -> None:
    # Use split parts so GitHub secret scanning scanner does not flag test fixture strings
    slack_dummy = "xoxb-" + "000000000000-" + "0000000000000-" + "abcdef123456789012345678"
    google_dummy = "AIzaSy" + "B1234567890abcdefghijklmnopqrstuv"
    text = f"Google {google_dummy} Slack {slack_dummy}"
    cleaned = sanitize_string(text)
    assert "[REDACTED_GOOGLE_API_KEY]" in cleaned
    assert "[REDACTED_SLACK_TOKEN]" in cleaned


def test_sanitize_pii_emails_phones_and_ips() -> None:
    text = "Contact alice.smith@enterprise.org at +1 (555) 234-5678 or 192.168.1.105 with card 4111 2222 3333 4444"
    cleaned = sanitize_string(text)
    assert "[REDACTED_EMAIL]" in cleaned
    assert "alice.smith" not in cleaned
    assert "[REDACTED_PHONE]" in cleaned
    assert "(555) 234-5678" not in cleaned
    assert "[REDACTED_IP_ADDRESS]" in cleaned
    assert "[REDACTED_CREDIT_CARD]" in cleaned


def test_sanitize_headers() -> None:
    headers = {
        "Authorization": "Bearer secret_jwt_token_12345",
        "X-Api-Key": "my-secret-key",
        "Content-Type": "application/json",
        "User-Email": "dev@company.com",
    }
    cleaned = sanitize_headers(headers)
    assert cleaned["authorization"] == "[REDACTED]"
    assert cleaned["x-api-key"] == "[REDACTED]"
    assert cleaned["content-type"] == "application/json"
    assert cleaned["user-email"] == "[REDACTED_EMAIL]"


def test_sanitize_nested_payload() -> None:
    dummy_key = "sk-" + "12345678901234567890123456789012"
    payload = {
        "prompt": f"Here is key {dummy_key}",
        "nested": ["Bearer secret_token_xyz_12345", "contact@test.org"],
    }
    cleaned = sanitize_payload(payload)
    assert "[REDACTED_OPENAI_KEY]" in cleaned["prompt"]
    assert "Bearer [REDACTED_TOKEN]" in cleaned["nested"][0]
    assert "[REDACTED_EMAIL]" in cleaned["nested"][1]


def test_sanitize_with_custom_patterns() -> None:
    custom = [
        (re.compile(r"INTERNAL-[0-9]{4}"), "[REDACTED_INTERNAL_ID]"),
    ]
    text = "Project ticket is INTERNAL-9876"
    cleaned = sanitize_string(text, custom_patterns=custom)
    assert cleaned == "Project ticket is [REDACTED_INTERNAL_ID]"


def test_sanitize_url_sensitive_query_parameters() -> None:
    raw_url = "https://generativelanguage.googleapis.com/v1beta/models?key=AIzaSySecret123&client_secret=super_secret&token=tok_999&search=test_term"
    cleaned = sanitize_url(raw_url)
    assert "AIzaSySecret123" not in cleaned
    assert "super_secret" not in cleaned
    assert "tok_999" not in cleaned
    assert "key=%5BREDACTED%5D" in cleaned or "key=[REDACTED]" in cleaned
    assert "client_secret=%5BREDACTED%5D" in cleaned or "client_secret=[REDACTED]" in cleaned
    assert "search=test_term" in cleaned


SPLIT_SSE_CHUNKS = [
    'data: {"token":"mail alice.smith@exam',
    'ple.org done"}\n\n',
    "data: [DONE]\n\n",
]


def test_sanitize_chunks_redacts_values_split_across_chunks() -> None:
    cleaned = sanitize_chunks(SPLIT_SSE_CHUNKS)
    assert cleaned == [
        'data: {"token":"mail [REDACTED_EMAIL] done"}\n\n',
        "",
        "data: [DONE]\n\n",
    ]
    assert "".join(cleaned) == sanitize_string("".join(SPLIT_SSE_CHUNKS))


def test_sanitize_chunks_keeps_boundaries_when_nothing_straddles() -> None:
    chunks = [
        'data: {"token":"Hello"}\n\n',
        'data: {"token":"mail alice.smith@example.org"}\n\n',
        "data: [DONE]\n\n",
    ]
    assert sanitize_chunks(chunks) == [sanitize_string(chunk) for chunk in chunks]


def test_sanitize_chunks_at_every_split_point() -> None:
    text = (
        'data: {"email":"alice.smith@example.org","card":"4111 1111 1111 1111",'
        '"ip":"192.168.1.105","key":"sk-12345678901234567890123456789012"}\n\n'
    )
    expected = sanitize_string(text)
    assert "alice.smith" not in expected

    for index in range(1, len(text)):
        cleaned = sanitize_chunks([text[:index], text[index:]])
        assert len(cleaned) == 2
        assert "".join(cleaned) == expected

    for first in range(1, len(text) - 1, 5):
        for second in range(first + 1, len(text), 7):
            cleaned = sanitize_chunks([text[:first], text[first:second], text[second:]])
            assert len(cleaned) == 3
            assert "".join(cleaned) == expected


def test_sanitize_chunks_applies_custom_patterns_across_boundaries() -> None:
    custom = [(re.compile(r"INTERNAL-[0-9]{4}"), "[REDACTED_INTERNAL_ID]")]
    cleaned = sanitize_chunks(["ticket INTERNAL-98", "76 open"], custom_patterns=custom)
    assert cleaned == ["ticket [REDACTED_INTERNAL_ID] open", ""]
