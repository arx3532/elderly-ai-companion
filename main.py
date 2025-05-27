import logging
from typing import Dict, Any, TypedDict
from langgraph.graph import StateGraph, END
from agents.router_agent import router_agent
from agents.memory_agent import memory_agent
from agents.tone_agent import tone_agent
from agents.conversation_agent import conversation_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    user_input: str
    user_id: str
    user_profile: Dict[str, Any]
    response: str
    tone: str
    next_agent: str
    current_agent: str
    is_initial_retrieval: bool
    conversation_history: list

def setup_workflow():
    workflow = StateGraph(AgentState)
    workflow.add_node("router", router_agent)
    workflow.add_node("memory", memory_agent)
    workflow.add_node("tone_setter", tone_agent)
    workflow.add_node("conversation", conversation_agent)

    # Router decides which agent to call next
    workflow.add_conditional_edges(
        "router",
        lambda state: state["next_agent"],
        {
            "memory": "memory", 
            "tone": "tone_setter", 
            "conversation": "conversation", 
            "END": END
        }
    )
    
    # All agents return to router for next decision
    workflow.add_edge("memory", "router")
    workflow.add_edge("tone_setter", "router")
    workflow.add_edge("conversation", "router")

    workflow.set_entry_point("router")
    return workflow.compile()

app = setup_workflow()

def run_companion(user_input: str):
    state: AgentState = {
        "user_input": user_input,
        "user_id": "user_123",
        "user_profile": {},
        "response": "",
        "tone": "",
        "next_agent": "",
        "current_agent": "",
        "is_initial_retrieval": True,
        "conversation_history": []
    }
    
    logger.info(f"Initial state: {state}")
    
    try:
        result = app.invoke(state)
        logger.info(f"Final state: {result}")
        
        # Check if workflow completed successfully
        if result.get("next_agent") != "END":
            logger.error(f"Workflow failed to complete properly: {result}")
            raise RuntimeError("Workflow failed to complete")
            
        return result["user_profile"], result["tone"], result["response"]
        
    except Exception as e:
        logger.error(f"Workflow error: {e}")
        raise

if __name__ == "__main__":
    print("AI Companion: Hello! How can I help you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("AI Companion: Goodbye!")
            break
        try:
            profile, tone, response = run_companion(user_input)
            print(f"AI Companion: {response}")
        except Exception as e:
            print(f"AI Companion: Sorry, something went wrong: {e}")