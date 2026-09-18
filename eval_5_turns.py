import sys
import os
import json
import time
from pathlib import Path
import requests

# Ensure TOKEN_BUDGET_LIMIT is low enough to force eviction of older turns from 'recent' history
os.environ["TOKEN_BUDGET_LIMIT"] = "100"

BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
SESSION_ID = "eval_session_5turn"
LOG_FILE = Path(f"./logs/context_{SESSION_ID}.json")

def send_chat_request(session_id: str, message: str) -> dict:
    """Sends POST request to /v1/chat endpoint and returns JSON response."""
    url = f"{BASE_URL}/v1/chat"
    payload = {"session_id": session_id, "message": message}
    
    try:
        res = requests.post(url, json=payload, timeout=10)
        if res.status_code != 200:
            print(f"Error: API returned status code {res.status_code}: {res.text}")
            sys.exit(1)
        return res.json()
    except requests.exceptions.ConnectionError:
        # Fallback to FastAPI TestClient if live server is not running on localhost:8000
        from fastapi.testclient import TestClient
        from src.api import app
        
        client = TestClient(app)
        res = client.post("/v1/chat", json=payload)
        if res.status_code != 200:
            print(f"TestClient Error: status code {res.status_code}: {res.text}")
            sys.exit(1)
        return res.json()

def main():
    print("=" * 60)
    print("Starting 5-Turn Context Engineering Evaluation")
    print("=" * 60)
    
    # Remove existing session log file and in-memory session to ensure clean evaluation state
    if LOG_FILE.exists():
        try:
            LOG_FILE.unlink()
        except Exception:
            pass
            
    from src.context_engine import SESSION_STORE
    SESSION_STORE.pop(SESSION_ID, None)
    
    # Turn 1: Greet
    print("\n[Turn 1] User: 'Hi, I need help.'")
    res1 = send_chat_request(SESSION_ID, "Hi, I need help.")
    print(f"Agent Response: {res1.get('response')}")
    print(f"Active Topic: {res1.get('active_topic')}")
    assert res1.get("active_topic") == "general", f"Expected topic 'general', got '{res1.get('active_topic')}'"
    
    # Turn 2: Provide Ticket ID
    print("\n[Turn 2] User: 'My ticket is IT-9921.'")
    res2 = send_chat_request(SESSION_ID, "My ticket is IT-9921.")
    print(f"Agent Response: {res2.get('response')}")
    print(f"Active Topic: {res2.get('active_topic')}")
    assert "ticket" in res2.get("active_topic", "").lower(), f"Expected ticket topic, got '{res2.get('active_topic')}'"
    
    # Verify pinned.ticket_id after Turn 2
    assert LOG_FILE.exists(), f"Context log file {LOG_FILE} not found!"
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        ctx2 = json.load(f)
    assert ctx2.get("pinned", {}).get("ticket_id") == "IT-9921", f"Expected pinned ticket_id 'IT-9921', got '{ctx2.get('pinned', {}).get('ticket_id')}'"
    print("-> Verified: Ticket ID 'IT-9921' stored in pinned memory.")
    
    # Turn 3: Ask a long question about Mac WiFi (triggers topic switch & token eviction)
    turn3_msg = (
        "Actually, before we do that, my WiFi in the library just dropped. "
        "I am using a Mac computer. It says 'connected without internet' and I cannot load any websites. "
        "Could you please guide me on how to fix this campus network connection issue?"
    )
    print("\n[Turn 3] User: Ask long Mac WiFi question (forcing token eviction)")
    res3 = send_chat_request(SESSION_ID, turn3_msg)
    print(f"Agent Response: {res3.get('response')}")
    print(f"Active Topic: {res3.get('active_topic')}")
    assert "wifi" in res3.get("active_topic", "").lower(), f"Expected wifi topic, got '{res3.get('active_topic')}'"
    
    # Turn 4: Ask follow-up details on WiFi
    print("\n[Turn 4] User: 'Can you give me the Mac troubleshooting steps?'")
    res4 = send_chat_request(SESSION_ID, "Can you give me the Mac troubleshooting steps?")
    print(f"Agent Response: {res4.get('response')}")
    print(f"Active Topic: {res4.get('active_topic')}")
    
    # Turn 5: Return to ticket inquiry
    print("\n[Turn 5] User: 'Okay, I fixed the WiFi. Let's go back to my original ticket. Can you confirm the status of the ticket I mentioned earlier?'")
    res5 = send_chat_request(
        SESSION_ID,
        "Okay, I fixed the WiFi. Let's go back to my original ticket. Can you confirm the status of the ticket I mentioned earlier?"
    )
    reply_5 = res5.get("response", "")
    print(f"Agent Response: {reply_5}")
    print(f"Active Topic: {res5.get('active_topic')}")
    
    # Perform strict evaluation assertions on context file and turn 5 LLM response
    print("\n" + "=" * 60)
    print("Running Final Evaluation Assertions")
    print("=" * 60)
    
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        final_context = json.load(f)
        
    pinned_ticket = final_context.get("pinned", {}).get("ticket_id")
    recent_messages = final_context.get("recent", [])
    
    print(f"Final Pinned Ticket ID: {pinned_ticket}")
    print(f"Total Recent Messages Count: {len(recent_messages)}")
    print(f"Total Recent Tokens: {final_context.get('total_recent_tokens')}")
    
    # Assertion 1: pinned.ticket_id MUST equal IT-9921
    assert pinned_ticket == "IT-9921", f"FAIL: Expected pinned.ticket_id == 'IT-9921', got '{pinned_ticket}'"
    print("[PASS] Assertion 1: pinned.ticket_id == 'IT-9921'")
    
    # Assertion 2: Turn 2 ticket message was evicted from 'recent' array due to token budget limit
    recent_user_texts = [msg.get("content", "") for msg in recent_messages if msg.get("role") == "user"]
    turn_2_user_present = any("IT-9921" in t for t in recent_user_texts)
    assert not turn_2_user_present, "FAIL: Turn 2 user message containing IT-9921 should have been evicted from recent array!"
    print("[PASS] Assertion 2: Turn 2 message evicted from 'recent' rolling window array.")
    
    # Assertion 3: Final LLM response text MUST contain "IT-9921"
    assert "IT-9921" in reply_5, f"FAIL: LLM reply does not contain ticket ID 'IT-9921'. Reply was: {reply_5}"
    print("[PASS] Assertion 3: Final LLM response intelligently referenced 'IT-9921'.")
    
    print("\nSUCCESS: All 5-Turn Context Engineering evaluation assertions PASSED!")
    sys.exit(0)

if __name__ == "__main__":
    main()
