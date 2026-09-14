import os
import shutil
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from src.api import app
from src.context_engine import LOGS_DIR

client = TestClient(app)

@pytest.fixture(autouse=True)
def cleanup_logs():
    """Cleanup logs directory before and after tests."""
    yield
    if LOGS_DIR.exists():
        for f in LOGS_DIR.glob("context_*.json"):
            try:
                f.unlink()
            except Exception:
                pass

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_context_not_found():
    response = client.get("/v1/context/non_existent_session_999")
    assert response.status_code == 404

def test_chat_and_context_persistence():
    session_id = "test_session_123"
    
    # Send generic greeting
    res = client.post("/v1/chat", json={"session_id": session_id, "message": "Hello, I need help!"})
    assert res.status_code == 200
    data = res.json()
    assert "response" in data
    assert data["active_topic"] == "general"
    
    # Check context persistence endpoint
    ctx_res = client.get(f"/v1/context/{session_id}")
    assert ctx_res.status_code == 200
    ctx_data = ctx_res.json()
    assert ctx_data["session_id"] == session_id
    assert ctx_data["active_topic"] == "general"
    assert ctx_data["pinned"]["ticket_id"] is None
    assert len(ctx_data["recent"]) == 2  # user + assistant
    assert "total_recent_tokens" in ctx_data
    
    # Check file system log
    log_file = Path(f"./logs/context_{session_id}.json")
    assert log_file.exists()

def test_ticket_id_pinning():
    session_id = "ticket_test"
    res = client.post(
        "/v1/chat",
        json={"session_id": session_id, "message": "Hello, I am calling about my ticket IT-4921."}
    )
    assert res.status_code == 200
    
    ctx_res = client.get(f"/v1/context/{session_id}")
    assert ctx_res.status_code == 200
    ctx_data = ctx_res.json()
    assert ctx_data["pinned"]["ticket_id"] == "IT-4921"
    assert ctx_data["active_topic"] == "ticket_inquiry"

def test_token_budget_eviction():
    session_id = "token_eviction_test"
    os.environ["TOKEN_BUDGET_LIMIT"] = "50"
    
    # Send turn 1 (long message)
    msg_a = "Message A: " + ("Hello IT support team, I need urgent assistance with my campus network credentials. " * 3)
    res1 = client.post("/v1/chat", json={"session_id": session_id, "message": msg_a})
    assert res1.status_code == 200
    
    # Send turn 2 (another long message)
    msg_b = "Message B: " + ("Also my laptop refuses to connect to the eduroam wireless access point in the main hall. " * 3)
    res2 = client.post("/v1/chat", json={"session_id": session_id, "message": msg_b})
    assert res2.status_code == 200
    
    ctx_res = client.get(f"/v1/context/{session_id}")
    assert ctx_res.status_code == 200
    ctx_data = ctx_res.json()
    
    # Total tokens in recent must be <= TOKEN_BUDGET_LIMIT (50)
    assert ctx_data["total_recent_tokens"] <= 50

def test_atomic_error_handling():
    session_id = "error_test_1"
    os.environ["OPENAI_API_KEY"] = "invalid_key_for_testing_500"
    
    # Request should fail with 500
    res = client.post("/v1/chat", json={"session_id": session_id, "message": "This message should fail."})
    assert res.status_code == 500
    
    # Context should NOT exist or be created for failed session
    ctx_res = client.get(f"/v1/context/{session_id}")
    assert ctx_res.status_code == 404
    
    # Reset API key
    os.environ["OPENAI_API_KEY"] = "mock_key_for_testing"
