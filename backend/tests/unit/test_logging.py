import logging

from app.infrastructure.observability.logging import KeyValueFormatter


def render(**extra):
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "msg", (), None)
    record.__dict__.update(extra)
    return KeyValueFormatter("%(message)s").format(record)


def test_secrets_are_masked_but_token_counts_are_not():
    line = render(api_key="sk-live", admin_token="abc", input_tokens=713, output_tokens=366)
    assert "sk-live" not in line and "abc" not in line
    assert "input_tokens=713" in line and "output_tokens=366" in line
