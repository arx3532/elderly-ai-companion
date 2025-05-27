import logging
import json
from typing import Dict, Any
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def router_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    state.setdefault("user_input", "")
    state.setdefault("current_agent", "")
    state.setdefault("is_initial_retrieval", True)
    state.setdefault("user_profile", {})
    state.setdefault("tone", "")
    state.setdefault("response", "")
    
    user_input = state["user_input"].strip()
    current_agent = state["current_agent"]
    is_initial_retrieval = state["is_initial_retrieval"]
    has_profile = bool(state["user_profile"])
    tone_set = state["tone"] != ""
    response_exists = state["response"] != ""
    
    logger.info(f"Router Agent: Processing input: {user_input}, current_agent: {current_agent}")
    
    def determine_next_agent():
        if is_initial_retrieval and not has_profile:
            return "memory"
        if any(keyword in user_input.lower() for keyword in ["about me", "my profile", "fun fact", "tell me about myself"]):
            return "memory" if is_initial_retrieval else ("conversation" if tone_set else "memory")
        if has_profile and not tone_set:
            return "tone"
        if has_profile and tone_set and not response_exists:
            return "conversation"
        if response_exists and current_agent != "memory":
            return "memory"
        return "END"

    use_llm = state.get("use_llm_for_routing", True)
    next_agent = None
    
    if use_llm:
        llm = ChatOllama(model='gemma3:4b', base_url="http://localhost:11434", temperature=0.7)
        prompt = f"""
You are a router for an AI companion workflow. Choose the next agent based on state and input. Agents:
- memory: Retrieves/stores profile data.
- tone: Sets response tone.
- conversation: Generates response.
- END: Ends workflow.

Rules:
1. If is_initial_retrieval=True and has_profile=False → memory.
2. If has_profile=True and tone_set=False → tone.
3. If has_profile=True and tone_set=True and response_exists=False → conversation.
4. If response_exists=True and current_agent != "memory" → memory.
5. If response_exists=True and current_agent="memory" → END.
6. For profile queries (e.g., "fun fact about me"), use memory if is_initial_retrieval=True, else conversation if tone_set=True.

State:
- is_initial_retrieval: {is_initial_retrieval}
- has_profile: {has_profile}
- tone_set: {tone_set}
- current_agent: {current_agent}
- response_exists: {response_exists}
- user_input: {user_input}

Return a JSON object with a single key "next_agent" containing the routing decision.
"""
        try:
            llm_output = llm.invoke([SystemMessage(content=prompt)]).content.strip()
            logger.info(f"Router Agent: Raw LLM output: {llm_output}")
            llm_output_clean = llm_output.replace('<think>', '').replace('</think>', '').strip()
            try:
                routing_decision = json.loads(llm_output_clean)
                next_agent = routing_decision.get("next_agent")
            except json.JSONDecodeError:
                logger.warning("LLM output is not valid JSON, attempting fallback detection")
                match = re.search(r'\b(memory|tone|conversation|END)\b', llm_output_clean, re.IGNORECASE)
                if match:
                    next_agent = match.group(1).lower()
        except Exception as e:
            logger.error(f"LLM routing failed: {e}")
    
    if next_agent not in ["memory", "tone", "conversation", "END"]:
        next_agent = determine_next_agent()
        logger.info(f"Router Agent: Using fallback routing → {next_agent}")
    
    state["next_agent"] = next_agent
    state["current_agent"] = next_agent
    return state
