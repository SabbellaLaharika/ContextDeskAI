import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from src.context_engine import (
    AgentContext,
    Message,
    PinnedFacts,
    load_context,
    save_context
)
from src.extractor import extract_ticket_id, extract_created_ticket_id, infer_active_topic
from src.token_budget import apply_token_budget
from src.llm_client import generate_llm_response

load_dotenv()

app = FastAPI(
    title="Context-Aware IT Helpdesk Agent API",
    version="1.0.0"
)

STATIC_UI_FILE = Path(__file__).parent / "static" / "index.html"

from typing import Optional

class ChatRequest(BaseModel):
    session_id: str
    message: str
    token_budget_limit: Optional[int] = None

class ChatResponse(BaseModel):
    response: str
    active_topic: str

def get_token_budget_limit() -> int:
    """Reads TOKEN_BUDGET_LIMIT environment variable, default 4000."""
    try:
        return int(os.getenv("TOKEN_BUDGET_LIMIT", "4000"))
    except ValueError:
        return 4000

@app.get("/", response_class=FileResponse)
@app.get("/ui", response_class=FileResponse)
async def serve_ui():
    """Serves the interactive Web UI for context engineering testing."""
    if STATIC_UI_FILE.exists():
        return FileResponse(STATIC_UI_FILE)
    raise HTTPException(status_code=404, detail="UI index.html file not found.")

@app.get("/health")
async def health_check():
    """Health check endpoint required by docker-compose and evaluation."""
    return {"status": "ok"}

@app.get("/v1/context/{session_id}")
async def get_context(session_id: str):
    """
    Retrieves the current state of a session's context.
    Returns HTTP 404 if session_id does not exist.
    """
    context = load_context(session_id, create_if_missing=False)
    if not context:
        raise HTTPException(
            status_code=404,
            detail=f"Session ID '{session_id}' not found."
        )
    return JSONResponse(content=context.to_dict(), status_code=200)

@app.post("/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Conversational API endpoint.
    Processes user message, updates pinned facts & active topic,
    enforces token budget, calls LLM, and persists context atomically.
    """
    if not req.session_id or not req.message:
        raise HTTPException(status_code=400, detail="session_id and message are required.")

    token_limit = req.token_budget_limit if req.token_budget_limit is not None else get_token_budget_limit()

    # Load or create session context
    original_context = load_context(req.session_id, create_if_missing=True)
    assert original_context is not None

    # Staging changes: Work on a temporary copy to guarantee atomic state rollback if LLM fails
    new_ticket_id = extract_ticket_id(req.message)
    staged_ticket_id = new_ticket_id or original_context.pinned.ticket_id
    staged_topic = infer_active_topic(req.message, original_context.active_topic)

    # Staged context for LLM system prompt construction
    staged_context = AgentContext(
        session_id=req.session_id,
        active_topic=staged_topic,
        pinned=PinnedFacts(ticket_id=staged_ticket_id),
        recent=list(original_context.recent),
        total_recent_tokens=original_context.total_recent_tokens
    )

    # Stage incoming user message
    user_msg = Message(role="user", content=req.message)
    staged_recent = staged_context.recent + [user_msg]

    # Apply token budget eviction to user message history before calling upstream LLM
    pruned_staged_recent, _ = apply_token_budget(staged_recent, token_limit)

    # Execute upstream LLM generation call
    try:
        reply_text = generate_llm_response(staged_context, pruned_staged_recent)
    except Exception as e:
        # Atomic Rollback (Req 10): If LLM fails, return HTTP 500 and DO NOT persist staged user message
        raise HTTPException(
            status_code=500,
            detail=f"Upstream LLM error: {str(e)}"
        )

    # LLM generation succeeded: stage assistant response
    assistant_msg = Message(role="assistant", content=reply_text)
    final_recent = pruned_staged_recent + [assistant_msg]

    # If staged_ticket_id was not provided by user message, check if LLM generated a new Ticket ID
    if not staged_ticket_id:
        llm_generated_ticket = extract_created_ticket_id(reply_text)
        if llm_generated_ticket:
            staged_ticket_id = llm_generated_ticket

    # Re-apply token budget limit to ensure recent array with assistant response remains under limit
    pruned_final_recent, final_token_count = apply_token_budget(final_recent, token_limit)

    # Commit staged mutations to original context
    original_context.pinned.ticket_id = staged_ticket_id
    original_context.active_topic = staged_topic
    original_context.recent = pruned_final_recent
    original_context.total_recent_tokens = final_token_count

    # Persist state to ./logs/context_{session_id}.json
    save_context(original_context)

    return ChatResponse(
        response=reply_text,
        active_topic=original_context.active_topic
    )
