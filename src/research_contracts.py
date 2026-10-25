"""Reusable structured-output contracts for synthetic research evaluation.

These schemas are application-neutral. They model public-discussion research
outputs without containing LinkedIn credentials, messages, or application code.
"""

from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError


POST_CLASSIFICATION_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "relevance_score",
        "category",
        "reasoning_summary",
        "additional_research_justified",
    ],
    "properties": {
        "relevance_score": {"type": "integer", "minimum": 0, "maximum": 100},
        "category": {
            "type": "string",
            "enum": [
                "ai_engineering",
                "business_operations",
                "leadership",
                "marketing",
                "unrelated",
            ],
        },
        "reasoning_summary": {"type": "string", "minLength": 1, "maxLength": 500},
        "additional_research_justified": {"type": "boolean"},
    },
}


DISCUSSION_ANALYSIS_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "central_argument",
        "unanswered_questions",
        "discussion_quality",
        "potential_contribution_angle",
        "explicitly_supported_business_needs",
        "source_grounded",
    ],
    "properties": {
        "central_argument": {"type": "string", "minLength": 1, "maxLength": 800},
        "unanswered_questions": {
            "type": "array",
            "items": {"type": "string", "minLength": 1, "maxLength": 500},
            "maxItems": 8,
        },
        "discussion_quality": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
        "potential_contribution_angle": {
            "type": "string",
            "minLength": 1,
            "maxLength": 800,
        },
        "explicitly_supported_business_needs": {
            "type": "array",
            "items": {"type": "string", "minLength": 1, "maxLength": 300},
            "maxItems": 8,
        },
        "source_grounded": {"type": "boolean"},
    },
}


CONTRACTS = {
    "post_classification": POST_CLASSIFICATION_SCHEMA,
    "discussion_analysis": DISCUSSION_ANALYSIS_SCHEMA,
}


class StructuredOutputError(ValueError):
    """Raised when a requested structured response cannot be trusted."""


def checked_schema(response_format: object) -> dict[str, Any] | None:
    """Validate an OpenAI response_format and return its JSON schema, if any."""

    if response_format is None:
        return None
    if not isinstance(response_format, dict):
        raise StructuredOutputError("response_format must be an object")
    format_type = response_format.get("type")
    if format_type == "json_object":
        return {"type": "object"}
    if format_type != "json_schema":
        raise StructuredOutputError(
            "response_format.type must be json_object or json_schema"
        )
    wrapper = response_format.get("json_schema")
    if not isinstance(wrapper, dict):
        raise StructuredOutputError("response_format.json_schema must be an object")
    schema = wrapper.get("schema")
    if not isinstance(schema, dict):
        raise StructuredOutputError("response_format.json_schema.schema must be an object")
    if _node_count(schema) > 2048:
        raise StructuredOutputError("JSON schema is too complex")
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise StructuredOutputError("response_format contains an invalid JSON schema") from exc
    return schema


def validate_instance(value: object, schema: dict[str, Any]) -> None:
    """Validate model-produced data against an independently supplied schema."""

    try:
        Draft202012Validator(schema).validate(value)
    except ValidationError as exc:
        raise StructuredOutputError(
            f"Model output failed JSON schema validation at {list(exc.absolute_path)}"
        ) from exc


def _node_count(value: object) -> int:
    if isinstance(value, dict):
        return 1 + sum(_node_count(key) + _node_count(item) for key, item in value.items())
    if isinstance(value, list):
        return 1 + sum(_node_count(item) for item in value)
    return 1
