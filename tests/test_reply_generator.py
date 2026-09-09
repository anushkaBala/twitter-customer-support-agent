"""
Unit tests for reply generator.
"""

import pytest
from src.reply_generator import ReplyGenerator


@pytest.fixture
def generator():
    """Initialize reply generator for tests."""
    brand_config = {
        "voice": {
            "tone": "casual, helpful",
            "formality": "informal"
        },
        "policies": {
            "refund_window_days": 30,
            "standard_shipping_days": 5
        },
        "resolution_patterns": {
            "shipping_delay": ["Provide tracking update", "Offer expedited reshipping"],
            "refund_request": ["Confirm return shipment", "Process refund"]
        },
        "guardrails": []
    }
    return ReplyGenerator(brand="amazon", brand_config=brand_config)


def test_generate_reply_shipping_delay(generator):
    """Test reply generation for shipping delay."""
    result = generator.generate(
        message="My order is late",
        intent="shipping_delay",
        customer_history={}
    )
    
    assert "reply" in result
    assert len(result["reply"]) > 0
    assert isinstance(result.get("quality_score"), (int, float))
    assert 0 <= result["quality_score"] <= 1


def test_generate_reply_refund_request(generator):
    """Test reply generation for refund request."""
    result = generator.generate(
        message="I want a refund",
        intent="refund_request",
        customer_history={"order_value": 50}
    )
    
    assert "reply" in result
    assert len(result["reply"]) > 10
    assert "grounding" in result


def test_reply_guardrails(generator):
    """Test that guardrails prevent problematic replies."""
    result = generator.generate(
        message="When will it arrive?",
        intent="shipping_delay",
        customer_history={}
    )
    
    reply = result["reply"]
    # Should not guarantee specific dates
    assert "will definitely" not in reply.lower() or "will arrive" in reply.lower()


def test_reply_length_limit(generator):
    """Test that replies don't exceed max length."""
    result = generator.generate(
        message="Tell me everything about your policies",
        intent="general_inquiry",
        customer_history={}
    )
    
    assert len(result["reply"]) <= 500


def test_quality_score_estimation(generator):
    """Test quality score is reasonable."""
    result = generator.generate(
        message="Where's my order?",
        intent="order_status",
        customer_history={}
    )
    
    score = result.get("quality_score", 0.5)
    assert 0.3 <= score <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
