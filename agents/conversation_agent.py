import logging
import json
import re
from typing import Dict, Any
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def conversation_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates a response based on user input, profile, and tone.
    Args:
        state: AgentState with user input, profile, tone, and context.
    Returns:
        Updated state with response and conversation history.
    """
    user_input = state.get("user_input", "").strip()  # Don't convert to lowercase
    user_profile = state.get("user_profile", {})
    tone = state.get("tone", "neutral")
    conversation_history = state.get("conversation_history", [])

    logger.info(f"Conversation Agent: Input state user_profile: {user_profile}")

    llm = ChatOllama(model='gemma3:4b', base_url="http://localhost:11434", temperature=0.7)

    profile_data = user_profile.get("data", {})
    name = profile_data.get("name", "Friend")
    interests = profile_data.get("interests", [])
    emotional_state = profile_data.get("emotional_state", "neutral")

    # Simplified and clearer prompt
    prompt = f"""You are a friendly AI companion for elderly users. Generate a warm, personalized response to the user's input only upto 50 words.

Tone: {tone}
User's Name: {name}
User's Interests: {', '.join(interests) if interests else 'None specified'}
User's Emotional State: {emotional_state}
Recent Conversation: {conversation_history[-2:] if conversation_history else 'None'}

User Input: "{user_input}"

Instructions:
- Use a {tone} tone
- Address the user by name when appropriate
- Reference their interests or emotional state when relevant
- For questions about themselves (personality, fun facts, hobbies), use their profile data
- Keep responses conversational and engaging

Respond with ONLY a JSON object in this exact format:
{{"response": "Your response here"}}"""

    try:
        llm_output = llm.invoke([SystemMessage(content=prompt)]).content.strip()
        logger.info(f"Conversation Agent: Raw LLM output: {llm_output}")

        # Multiple JSON extraction strategies
        response = None
        
        # Strategy 1: Look for JSON code block
        json_block_match = re.search(r'```json\s*\n?(.*?)\n?```', llm_output, re.DOTALL)
        if json_block_match:
            try:
                response_data = json.loads(json_block_match.group(1).strip())
                response = response_data.get("response")
                logger.info("Successfully parsed JSON from code block")
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON from code block")

        # Strategy 2: Look for any JSON object with "response" key
        if not response:
            json_match = re.search(r'\{[^}]*"response"[^}]*\}', llm_output, re.DOTALL)
            if json_match:
                try:
                    response_data = json.loads(json_match.group(0))
                    response = response_data.get("response")
                    logger.info("Successfully parsed JSON object")
                except json.JSONDecodeError:
                    logger.warning("Failed to parse JSON object")

        # Strategy 3: Look for response value after "response": 
        if not response:
            response_match = re.search(r'"response":\s*"([^"]*)"', llm_output)
            if response_match:
                response = response_match.group(1)
                logger.info("Extracted response from key-value pair")

        # Strategy 4: Use entire output if it looks like a direct response
        if not response:
            # Clean up any JSON artifacts and use as direct response
            cleaned_output = re.sub(r'[{}"]', '', llm_output).strip()
            cleaned_output = re.sub(r'response:\s*', '', cleaned_output)
            if len(cleaned_output) > 10 and not cleaned_output.startswith('{'):
                response = cleaned_output
                logger.info("Using cleaned output as direct response")

        # Final fallback
        if not response:
            logger.warning("All JSON parsing strategies failed, using fallback")
            response = "I'd be happy to chat with you! Could you tell me a bit more about what you'd like to talk about?"

        # Update conversation history
        conversation_entry = f"User: {user_input} | AI: {response}"
        conversation_history.append(conversation_entry)
        
        # Keep only last 10 conversations to prevent memory bloat
        if len(conversation_history) > 10:
            conversation_history = conversation_history[-10:]

        # Update state
        state["response"] = response
        state["conversation_history"] = conversation_history
        
        # Also update the profile's conversation history for consistency
        if "conversation_history" not in state["user_profile"]:
            state["user_profile"]["conversation_history"] = []
        state["user_profile"]["conversation_history"] = conversation_history
        
        state["next_agent"] = "memory"  # Go to memory to store the conversation

        logger.info(f"Conversation Agent: Generated response: {response}")
        return state
        
    except Exception as e:
        logger.error(f"Conversation Agent failed: {str(e)}")
        
        # Fallback response
        fallback_responses = [
            f"Hello {name}! I'd love to chat with you about that.",
            "That's interesting! Could you tell me more?",
            "I enjoy our conversations! What would you like to talk about?",
            "Thanks for sharing that with me. How are you feeling today?"
        ]
        
        response = fallback_responses[0]  # Simple fallback
        
        conversation_entry = f"User: {user_input} | AI: {response}"
        conversation_history.append(conversation_entry)
        
        state["response"] = response
        state["conversation_history"] = conversation_history
        state["user_profile"]["conversation_history"] = conversation_history
        state["next_agent"] = "memory"
        
        logger.info(f"Conversation Agent: Used fallback response: {response}")
        return state