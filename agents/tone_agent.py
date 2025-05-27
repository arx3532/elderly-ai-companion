from typing import Dict, Any
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load sentiment model and tokenizer
try:
    tokenizer = DistilBertTokenizer.from_pretrained("./models/sentiment-model")
    model = DistilBertForSequenceClassification.from_pretrained("./models/sentiment-model")
    model.eval()  # Set to evaluation mode
except Exception as e:
    logger.error(f"Failed to load sentiment model: {e}")
    raise

def tone_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Determines the conversation tone using a pre-trained sentiment model with probabilities from 0 to 1.
    Args:
        state: AgentState containing user input and profile.
    Returns:
        Updated state with tone.
    """
    logger.info(f"Tone Agent: Input state user_profile: {state.get('user_profile')}")
    user_input = state.get("user_input", "").strip()
    profile = state.get("user_profile", {})
    emotional_history = profile.get("emotional_history", [])

    if not user_input:
        logger.warning("No user input provided, defaulting to neutral tone")
        state["tone"] = "neutral"
        logger.info(f"Tone Agent: Output state user_profile: {state.get('user_profile')}")
        return state

    # Tokenize input
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Run sentiment model
    try:
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).numpy()[0]
        logger.info(f"Sentiment model probabilities: {probabilities}")
    except Exception as e:
        logger.warning(f"Sentiment model inference failed: {e}, defaulting to neutral tone")
        state["tone"] = "neutral"
        logger.info(f"Tone Agent: Output state user_profile: {state.get('user_profile')}")
        return state

    # Use positive probability (index 1) for sentiment score
    sentiment_score = probabilities[1]  # Positive probability (0 to 1)
    logger.info(f"Sentiment score (positive probability): {sentiment_score}")

    # Map sentiment to tone
    if sentiment_score > 0.65:
        tone = "playful"
    elif sentiment_score < 0.35:
        tone = "empathetic"
    else:
        tone = "neutral"

    # Adjust tone based on emotional history
    if any("lonely" in history.lower() for history in emotional_history):
        tone = "empathetic"
    elif any("playful" in history.lower() for history in emotional_history):
        tone = "playful"

    state["tone"] = tone
    logger.info(f"Tone Agent: Set tone to {tone} for input: {user_input}")
    logger.info(f"Tone Agent: Output state user_profile: {state.get('user_profile')}")
    return state