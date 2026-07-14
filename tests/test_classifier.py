import json

import ticket_classifier


def test_billing_ticket():
    result = ticket_classifier.classify_support_ticket(
        "invoice is wrong",
        llm_classify_ticket=lambda ticket_text: json.dumps({"category": "billing"}),
    )

    assert result.category == "billing"
    assert result.valid is True
    assert result.error is None


def test_shipping_ticket():
    result = ticket_classifier.classify_support_ticket(
        "Where is my package?",
        llm_classify_ticket=lambda ticket_text: json.dumps({"category": "shipping"}),
    )

    assert result.category == "shipping"
    assert result.valid is True
    assert result.error is None

def test_technical_ticket():
    result = ticket_classifier.classify_support_ticket(
        "look at this bug in your software",
        llm_classify_ticket=lambda ticket_text: json.dumps({"category": "technical"}),
    )

    assert result.category == "technical"
    assert result.valid is True
    assert result.error is None

def test_account_ticket():
    result = ticket_classifier.classify_support_ticket(
        "I forgot my password , please help me reset it.",
        llm_classify_ticket=lambda ticket_text: json.dumps({"category": "account"}),
    )

    assert result.category == "account"
    assert result.valid is True
    assert result.error is None
    
def test_malformed_json_returns_other():
    result = ticket_classifier.classify_support_ticket(
        "malformed json test",
        llm_classify_ticket=lambda ticket_text: '{"category": "billing"',
    )

    assert result.category == "other"
    assert result.valid is False
    assert result.error == "malformed_json"


def test_invalid_category_returns_other():
    result = ticket_classifier.classify_support_ticket(
        "I want a refund.",
        llm_classify_ticket=lambda ticket_text: json.dumps(
            {"category": "refund_request"}
        ),
    )

    assert result.category == "other"
    assert result.valid is False
    assert result.error == "invalid_category"


def test_empty_ticket_returns_other_without_calling_llm():
    def fake_llm(ticket_text):
        raise AssertionError("LLM should not be called for empty tickets")

    result = ticket_classifier.classify_support_ticket(
        "   ",
        llm_classify_ticket=fake_llm,
    )

    assert result.category == "other"
    assert result.valid is True
    assert result.error == "empty_ticket"


def test_adversarial_ticket_with_two_possible_categories():
    result = ticket_classifier.classify_support_ticket(
        "I cannot access my account and I was charged twice.",
        llm_classify_ticket=lambda ticket_text: json.dumps({"category": "billing"}),
    )

    assert result.category == "billing"
    assert result.valid is True
    assert result.error is None




def test_real_stub_does_not_crash_over_many_calls():
    for _ in range(100):
        result = ticket_classifier.classify_support_ticket(
            "I need help with my invoice."
        )

        assert result.category in ticket_classifier.ALLOWED_CATEGORIES



