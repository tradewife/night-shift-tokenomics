"""Tests for data ingestion and seed manifest."""

from night_shift.data.ingestion.bootstrap import bootstrap_token_history
from night_shift.data.seed_manifest import list_seed_tokens, load_seed_manifest
from night_shift.taxonomy.parameter_space import DEFAULT_PARAMS


def test_seed_manifest_loads():
    manifest = load_seed_manifest()
    tokens = manifest["tokens"]
    assert len(tokens) >= 50
    labels = {t["label"] for t in tokens}
    assert "resilient" in labels
    assert "fragile" in labels


def test_bootstrap_label_differentiation():
    resilient = bootstrap_token_history(
        {"symbol": "JUP", "mint": "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN", "label": "resilient"}
    )
    fragile = bootstrap_token_history(
        {"symbol": "BONK", "mint": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263", "label": "fragile"}
    )
    assert resilient["top10_holder_pct"].mean() < fragile["top10_holder_pct"].mean()
    assert resilient["holder_count"].iloc[-1] > fragile["holder_count"].iloc[-1]


def test_list_seed_tokens_limit():
    tokens = list_seed_tokens(limit=5)
    assert len(tokens) == 5