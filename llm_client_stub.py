"""
Mock LLM client for candidates without API keys.
Swap this for a real provider call if you have one — interface should stay the same.

Deliberately returns malformed/unexpected output ~15% of the time so your
parsing code has something real to defend against. Don't remove that noise.
"""

import random
import json

CATEGORIES = ["billing", "shipping", "technical", "account", "other"]


def classify_ticket(ticket_text: str) -> str:
    """
    Simulates an LLM call. Returns a JSON *string* like a real API would —
    you are responsible for parsing it, validating the category, and
    handling the cases where it's garbage.
    """
    roll = random.random()

    if roll < 0.08:
        # malformed JSON
        return '{"category": "billing"'
    if roll < 0.15:
        # hallucinated category not in the allowed set
        return json.dumps({"category": "refund_request"})

    # naive keyword heuristic standing in for a real model, plus noise
    text = ticket_text.lower()
    if "charge" in text or "invoice" in text or "refund" in text:
        cat = "billing"
    elif "deliver" in text or "package" in text or "tracking" in text:
        cat = "shipping"
    elif "error" in text or "bug" in text or "crash" in text or "login" in text:
        cat = "technical"
    elif "password" in text or "account" in text:
        cat = "account"
    else:
        cat = random.choice(CATEGORIES)

    return json.dumps({"category": cat})
