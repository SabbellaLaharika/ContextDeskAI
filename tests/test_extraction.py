import pytest
from src.extractor import extract_ticket_id, infer_active_topic

def test_extract_ticket_id_valid():
    assert extract_ticket_id("Hello, I am calling about my ticket IT-4921.") == "IT-4921"
    assert extract_ticket_id("My ticket is IT-9921.") == "IT-9921"
    assert extract_ticket_id("Checking IT-1234 status") == "IT-1234"
    assert extract_ticket_id("it-8812 lower case check") == "IT-8812"

def test_extract_ticket_id_invalid():
    assert extract_ticket_id("No ticket here") is None
    assert extract_ticket_id("IT-123") is None  # Only 3 digits
    assert extract_ticket_id("TICKET-1234") is None

def test_infer_active_topic_general():
    assert infer_active_topic("Hi, I need help.", "general") == "general"
    assert infer_active_topic("Hello there", "general") == "general"

def test_infer_active_topic_ticket():
    assert infer_active_topic("My ticket is IT-9921", "general") == "ticket_inquiry"
    assert infer_active_topic("What was the status of the ticket I mentioned earlier?", "wifi_support") == "ticket_inquiry"

def test_infer_active_topic_wifi():
    assert infer_active_topic("My WiFi in the library just dropped. I am using a Mac.", "general") == "wifi_support"
    assert infer_active_topic("Can you give me the Mac troubleshooting steps?", "wifi_support") == "wifi_support"
