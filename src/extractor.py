import re
from typing import Optional

def extract_ticket_id(text: str) -> Optional[str]:
    """
    Dynamically identifies IT Ticket IDs matching the format IT-XXXX (where X is a digit) in user messages.
    Returns uppercase ticket ID (e.g. 'IT-4921', 'IT-9921') or None.
    """
    if not text:
        return None
    match = re.search(r"\bIT-\d{4}\b", text, re.IGNORECASE)
    if match:
        return match.group(0).upper()
    return None

def infer_active_topic(text: str, current_topic: str = "general") -> str:
    """
    Dynamically infers and updates active_topic based on the user's focus.
    Categorizes conversation into distinct semantic strings:
    - 'ticket_inquiry' for ticket-related queries
    - 'wifi_support' for WiFi / network troubleshooting
    - 'general' for generic greetings or default state
    """
    if not text:
        return current_topic or "general"
        
    lower_text = text.lower()
    
    # Priority 1: Ticket inquiry (ticket ID present or ticket keyword)
    if extract_ticket_id(text) or "ticket" in lower_text:
        return "ticket_inquiry"
        
    # Priority 2: WiFi and network connectivity support
    if any(k in lower_text for k in ["wifi", "wi-fi", "internet", "network", "connect", "mac"]):
        return "wifi_support"
        
    # Priority 3: Generic greetings
    if any(k in lower_text for k in ["hi", "hello", "hey", "help", "greet"]):
        if current_topic == "general" or not current_topic:
            return "general"
            
    return current_topic or "general"
