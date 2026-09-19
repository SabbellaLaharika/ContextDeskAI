import re
from typing import Optional

def normalize_ticket_ids_in_text(text: str) -> str:
    """
    Finds non-standard IT ticket patterns with year prefixes (e.g. IT-2026-01234) or >4 digits (IT-12345)
    and converts them to strict 4-digit IT-XXXX format.
    """
    if not text:
        return text

    # Replace IT-YYYY-XXXX (e.g. IT-2026-01234 -> IT-1234)
    def replace_year_prefix(match):
        digits = match.group(1)
        four_digits = digits[-4:].zfill(4)
        return f"IT-{four_digits}"

    text = re.sub(r"\bIT-\d{4}-(\d+)\b", replace_year_prefix, text, flags=re.IGNORECASE)

    # Replace IT-XXXXX (>4 digits, e.g. IT-12345 -> IT-2345)
    def replace_long_digits(match):
        digits = match.group(1)
        four_digits = digits[-4:]
        return f"IT-{four_digits}"

    text = re.sub(r"\bIT-(\d{5,})\b", replace_long_digits, text, flags=re.IGNORECASE)

    return text


def extract_ticket_id(text: str) -> Optional[str]:
    """
    Dynamically identifies IT Ticket IDs matching the format IT-XXXX (where X is a digit) in user messages.
    Returns uppercase ticket ID (e.g. 'IT-4921', 'IT-9921') or None.
    """
    if not text:
        return None
    normalized = normalize_ticket_ids_in_text(text)
    match = re.search(r"\bIT-\d{4}\b", normalized, re.IGNORECASE)
    if match:
        return match.group(0).upper()
    return None

def extract_created_ticket_id(reply_text: str, user_message: str = "") -> Optional[str]:
    """
    Extracts a ticket ID from assistant response ONLY if it represents an actual created/issued ticket
    and NOT an example, question, or format sample. Uses a localized window around each ticket ID.
    """
    if not reply_text:
        return None
        
    normalized = normalize_ticket_ids_in_text(reply_text)
    matches = list(re.finditer(r"\bIT-\d{4}\b", normalized, re.IGNORECASE))
    if not matches:
        return None
        
    user_lower = user_message.lower() if user_message else ""
    user_requested_escalation = any(k in user_lower for k in ["escalate", "human", "agent", "create ticket", "open ticket", "someone from it", "it help", "help desk", "submit ticket", "issue ticket"])
    
    for match in matches:
        ticket_id = match.group(0).upper()
        start, end = match.span()
        
        # Localized window around the ticket ID (80 characters before and after)
        win_start = max(0, start - 80)
        win_end = min(len(normalized), end + 80)
        window = normalized[win_start:win_end].lower()
        
        # 1. Skip if the immediate window contains example indicators
        example_indicators = ["for example", "e.g.", "sample", "something like", "such as", "like it-", "format"]
        if any(ind in window for ind in example_indicators):
            continue
            
        # 2. Check for explicit creation action verbs/phrases in the window
        creation_action_indicators = [
            "created", "generated", "opened", "issued", "assigned", 
            "your ticket", "your new ticket", "your support ticket", "ticket has been"
        ]
        has_creation_action = any(ind in window for ind in creation_action_indicators)
        
        # Return ticket_id if the assistant explicitly performed a creation action OR user requested escalation
        if has_creation_action or user_requested_escalation:
            return ticket_id
            
    return None




def infer_active_topic(text: str, current_topic: str = "general") -> str:
    """
    Dynamically infers and updates active_topic based on the user's focus.
    Categorizes conversation into distinct semantic strings:
    - 'ticket_inquiry' for ticket-related queries
    - 'password_reset' for password reset & account lockout support
    - 'printer_support' for printer & print queue support
    - 'vpn_support' for VPN & remote access support
    - 'software_support' for software installation & email setup
    - 'wifi_support' for WiFi / network troubleshooting
    - 'general' for generic greetings or default state
    """
    if not text:
        return current_topic or "general"
        
    lower_text = text.lower()
    
    # Priority 1: Ticket inquiry (ticket ID present or ticket keyword)
    if extract_ticket_id(text) or "ticket" in lower_text:
        return "ticket_inquiry"

    # Priority 2: Password Resets & Account Lockouts
    if any(k in lower_text for k in ["password", "lockout", "locked", "reset", "portal"]):
        return "password_reset"

    # Priority 3: Printer & Print Queue Resources
    if any(k in lower_text for k in ["print", "printer", "paper", "jam"]):
        return "printer_support"

    # Priority 4: VPN & Remote Access
    if any(k in lower_text for k in ["vpn", "remote access", "anyconnect", "globalprotect"]):
        return "vpn_support"

    # Priority 5: Software & Email Setup
    if any(k in lower_text for k in ["office", "office 365", "outlook", "software", "license"]):
        return "software_support"
        
    # Priority 6: WiFi and network connectivity support
    if any(k in lower_text for k in ["wifi", "wi-fi", "internet", "network", "connect", "mac"]):
        return "wifi_support"
        
    # Priority 7: Generic greetings
    if any(k in lower_text for k in ["hi", "hello", "hey", "help", "greet"]):
        if current_topic == "general" or not current_topic:
            return "general"
            
    return current_topic or "general"

