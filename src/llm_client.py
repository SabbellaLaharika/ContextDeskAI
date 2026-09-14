import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
from src.context_engine import AgentContext, Message

def build_system_prompt(context: AgentContext) -> str:
    """
    Constructs system prompt combining persona, active_topic, and pinned memory facts.
    Pinned facts and system prompt do NOT count towards TOKEN_BUDGET_LIMIT.
    """
    prompt_lines = [
        "You are an intelligent IT Helpdesk AI Assistant for a university campus.",
        "Your goal is to help students and faculty resolve technical issues clearly and efficiently.",
        f"Active Topic: {context.active_topic}"
    ]
    
    if context.pinned and context.pinned.ticket_id:
        prompt_lines.append(f"Pinned Fact - User Ticket ID: {context.pinned.ticket_id}")
    else:
        prompt_lines.append("Pinned Fact - User Ticket ID: None")
        
    prompt_lines.append(
        "Always reference the user's pinned ticket ID if they ask about their ticket or previous ticket status."
    )
    
    return "\n".join(prompt_lines)

def generate_llm_response(context: AgentContext, pruned_recent: List[Message]) -> str:
    """
    Executes upstream LLM call using OpenAI API.
    If the API call fails or key is invalid, raises an Exception to trigger atomic HTTP 500 rollback.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    # If explicit invalid test key provided for 500 test
    if api_key == "invalid_key_for_testing_500":
        raise RuntimeError("Invalid API key forced for error handling testing.")

    system_prompt = build_system_prompt(context)
    
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt}
    ]
    
    for msg in pruned_recent:
        messages.append({"role": msg.role, "content": msg.content})
        
    if not api_key or api_key == "mock_key_for_testing":
        # Mock LLM response generator for local unit testing when real API key is absent
        user_last = pruned_recent[-1].content if pruned_recent else ""
        ticket_id = context.pinned.ticket_id if context.pinned else None
        
        if ticket_id and ("status" in user_last.lower() or "ticket" in user_last.lower()):
            return f"I have checked your ticket {ticket_id}. The status is currently under review by IT support."
        elif "wifi" in user_last.lower() or "mac" in user_last.lower():
            return "To resolve Mac WiFi issues: 1. Forget the network, 2. Renew DHCP Lease under Network Settings, 3. Reconnect to Campus WiFi."
        elif ticket_id:
            return f"Thank you. I have recorded your ticket ID as {ticket_id}. How can I assist you with it today?"
        else:
            return "Hello! Welcome to the IT Helpdesk. How can I help you today?"
            
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore
        temperature=0.3
    )
    
    return response.choices[0].message.content or "I am unable to process your request."
