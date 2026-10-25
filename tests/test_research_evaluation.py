from src.config import SUPPORTED_MODELS
from src.research_evaluation import (
    load_cases,
    merge_trials,
    recommend_default,
    score_output,
)


def test_synthetic_cases_are_valid_and_task_balanced() -> None:
    cases = load_cases()
    assert len(cases) >= 4
    assert {case["task"] for case in cases} == {
        "post_classification",
        "discussion_analysis",
    }


def test_classification_scoring_uses_predefined_expected_result() -> None:
    case = load_cases()[0]
    output = {
        "relevance_score": 90,
        "category": "ai_engineering",
        "reasoning_summary": "Evaluation and latency evidence is present.",
        "additional_research_justified": True,
    }
    assert score_output(case, output) == 1.0


def test_recommendation_refuses_partial_model_comparison() -> None:
    result = recommend_default(
        [
            {
                "model": SUPPORTED_MODELS[0],
                "status": "measured",
                "quality": {"structured_output_success_rate": 1.0},
            }
        ]
    )
    assert result["status"] == "pending_complete_measurement"
    assert result["model"] is None


def test_recommendation_prioritizes_quality_then_latency() -> None:
    trials = []
    for index, model in enumerate(SUPPORTED_MODELS):
        trials.append(
            {
                "model": model,
                "context_length": 4096,
                "status": "measured",
                "quality": {
                    "structured_output_success_rate": 1.0,
                    "overall_quality": 0.8 + index * 0.05,
                    "median_latency_seconds": 3.0 - index,
                },
                "performance": {"median_wall_seconds": 2.0},
            }
        )
    result = recommend_default(trials)
    assert result["status"] == "measured_recommendation"
    assert result["model"] == SUPPORTED_MODELS[-1]


def test_sequential_report_merge_replaces_only_matching_trial() -> None:
    old = [
        {"model": "qwen3:14b", "context_length": 4096, "status": "old"},
        {"model": "gpt-oss:20b", "context_length": 4096, "status": "keep"},
    ]
    new = [{"model": "qwen3:14b", "context_length": 4096, "status": "measured"}]
    merged = merge_trials(old, new)
    assert len(merged) == 2
    assert next(item for item in merged if item["model"] == "qwen3:14b")["status"] == "measured"
    assert next(item for item in merged if item["model"] == "gpt-oss:20b")["status"] == "keep"
