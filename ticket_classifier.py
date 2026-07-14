import json
from collections.abc import Callable
from dataclasses import dataclass

from llm_client_stub import classify_ticket


ALLOWED_CATEGORIES = {"billing", "shipping", "technical", "account", "other"}


@dataclass
class TicketClassification:
    category: str
    valid: bool
    raw_response: str
    error: str | None = None


def parse_llm_response(raw_response: str) -> TicketClassification:
    """
    Parse and validate an LLM response.

    Bad model output should not crash the pipeline. Malformed JSON,
    missing fields, or invalid categories fall back to "other".
    """
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError:
        return TicketClassification(
            category="other",
            valid=False,
            raw_response=raw_response,
            error="malformed_json",
        )

    if not isinstance(parsed, dict):
        return TicketClassification(
            category="other",
            valid=False,
            raw_response=raw_response,
            error="invalid_json_shape",
        )

    category = parsed.get("category")
    if category not in ALLOWED_CATEGORIES:
        return TicketClassification(
            category="other",
            valid=False,
            raw_response=raw_response,
            error="invalid_category",
        )

    return TicketClassification(
        category=category,
        valid=True,
        raw_response=raw_response,
    )


def classify_support_ticket(
    ticket_text: str,
    llm_classify_ticket: Callable[[str], str] = classify_ticket,
) -> TicketClassification:
    """
    Classify a support ticket into one of the allowed routing categories.
    """
    if not ticket_text or not ticket_text.strip():
        return TicketClassification(
            category="other",
            valid=True,
            raw_response="",
            error="empty_ticket"
        )

    raw_response = llm_classify_ticket(ticket_text)
    return parse_llm_response(raw_response)
