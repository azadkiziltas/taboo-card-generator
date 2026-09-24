"""The benchmark metrics are computed correctly (fake provider, no LLM)."""

import csv
import json

from app.config.settings import BACKEND_ROOT
from app.domain.cards.entities import Difficulty
from tests.benchmark.runner import (
    BatchSpec,
    DatasetItem,
    load_dataset,
    parse_batch_spec,
    parse_model_key,
    run_batches,
    run_items,
    summarize,
    write_reports,
)
from tests.helpers import batch, build_env, card


def test_datasets_are_valid():
    for path in (BACKEND_ROOT / "tests/benchmark/datasets").glob("*.json"):
        items = load_dataset(path)
        assert items and all(i.target and len(i.language) == 2 for i in items)


def test_model_keys_keep_colons_in_model_names():
    assert parse_model_key("ollama:qwen3.5:27b") == ("ollama", "qwen3.5:27b")
    assert parse_batch_spec("tr:technology:5") == BatchSpec("tr", "technology", 5)
    assert parse_batch_spec("en::3") == BatchSpec("en", None, 3)


async def test_summary_metrics(tmp_path):
    env = build_env()
    env.primary.enqueue(
        "not json",  # parse failure -> retried
        batch(card("uçak", ["pilot", "kanat", "bilet", "uçuş", "havalimanı"])),
        batch(card("kahve", ["kahve", "fincan", "kafein", "çekirdek", "telve"])),  # invalid
        batch(card("kahve", ["sabah", "fincan", "kafein", "çekirdek", "telve"])),  # repaired
    )
    dataset = [
        DatasetItem("uçak", "tr", Difficulty.MEDIUM),
        DatasetItem("kahve", "tr", Difficulty.MEDIUM),
        DatasetItem("penguen", "tr", Difficulty.EASY),  # no scripted answer -> failure
    ]
    results = await run_items(env.container, "primary:model-a", dataset)
    summary = summarize("primary:model-a", results, env.usage)

    assert [r.success for r in results] == [True, True, False]
    assert results[2].error_type == "provider_unavailable"
    assert summary.final_success_rate == round(2 / 3, 4)
    assert summary.json_schema_success_rate == 0.8  # 1 parse failure out of 5 calls
    assert summary.first_pass_validation_rate == 0.5  # uçak ok, kahve needed a repair
    assert summary.retry_rate == round(2 / 3, 4)  # one parse retry + one repair
    assert summary.quality_score is None  # never invented automatically

    path = write_reports(tmp_path, [summary], results)
    report = json.loads((path / "report.json").read_text(encoding="utf-8"))
    assert report["summaries"][0]["model"] == "primary:model-a"
    with (path / "items.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["forbidden_words"] == "pilot|kanat|bilet|uçuş|havalimanı"
    assert "human_rating" in rows[0]


async def test_batch_metrics(tmp_path):
    env = build_env()
    env.primary.enqueue(batch(card("Kahve"), card("Robot")))
    batches = await run_batches(env.container, "primary:model-a", [BatchSpec("tr", "food", 3)])
    summary = summarize("primary:model-a", [], env.usage, batches=batches)
    assert batches[0].returned == 2 and batches[0].targets == ["Kahve", "Robot"]
    assert summary.batch_runs == 1 and summary.batch_return_rate == round(2 / 3, 4)
    path = write_reports(tmp_path, [summary], [], batches)
    assert (path / "batches.csv").read_text(encoding="utf-8").count("Kahve|Robot") == 1
