import logging
import json
import re
from typing import Dict, Any
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONVERSATION_BATCH_SIZE = 5

def memory_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    user_input = state.get("user_input", "").strip()
    user_id = state.get("user_id", "user_123")
    user_profile = state.get("user_profile", {})
    response = state.get("response", "")
    tone = state.get("tone", "neutral")
    conversation_history = state.get("conversation_history", [])
    is_initial_retrieval = state.get("is_initial_retrieval", True)
    pending_conversations = state.get("pending_conversations", [])

    logger.info(f"Memory Agent: Input state user_profile: {user_profile}, is_initial_retrieval: {is_initial_retrieval}")

    llm = ChatOllama(model='gemma3:4b', base_url="http://localhost:11434", temperature=0.1)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})

    try:
        vectorstore = Chroma(
            collection_name='user_profiles',
            embedding_function=embeddings,
            persist_directory='./chroma_db'
        )
    except Exception as e:
        logger.error(f"Failed to initialize Chroma: {e}")
        default_profile = {
            "user_id": user_id,
            "data": {"name": "Friend", "interests": [], "emotional_state": "neutral"},
            "conversation_history": conversation_history
        }
        state.update({"user_profile": default_profile, "is_initial_retrieval": False})
        return state

    default_profile = {
        "user_id": user_id,
        "data": {"name": "Aswin", "interests": ["washing clothes", "reading"], "emotional_state": "felt lonely 2 weeks back"},
        "conversation_history": conversation_history
    }

    def extract_json_response(text: str, fallback_data: dict) -> dict:
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        json_match = re.search(r'```json\s*\n?(.*?)\n?```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except json.JSONDecodeError:
                pass
        json_obj_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_obj_match:
            try:
                return json.loads(json_obj_match.group(0))
            except json.JSONDecodeError:
                pass
        return fallback_data

    if is_initial_retrieval or not user_profile:
        logger.info("Memory Agent: Retrieval mode")
        retrieval_prompt = f"""Generate a search query to find relevant user profile data.
User Input: "{user_input}"
Respond with ONLY a JSON object:
{{"query": "search terms here"}}"""
        try:
            retrieval_response = llm.invoke([SystemMessage(content=retrieval_prompt)]).content.strip()
            retrieval_data = extract_json_response(retrieval_response, {"query": user_input})
            query = retrieval_data.get("query", user_input)
            results = vectorstore.similarity_search_with_score(query, k=3)
            profile = default_profile.copy()
            for doc, score in results:
                try:
                    doc_data = json.loads(doc.page_content)
                    if doc_data.get("user_id") == user_id:
                        if "data" in doc_data:
                            profile["data"].update(doc_data["data"])
                        if "conversation_history" in doc_data:
                            profile["conversation_history"].extend(doc_data["conversation_history"])
                except json.JSONDecodeError:
                    continue
            profile["conversation_history"] = list(dict.fromkeys(profile["conversation_history"]))[-10:]
            state.update({"user_profile": profile, "conversation_history": profile["conversation_history"], "is_initial_retrieval": False})
            return state
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            state.update({"user_profile": default_profile, "is_initial_retrieval": False})
            return state

    elif response:
        logger.info("Memory Agent: Storage mode")
        conversation_entry = f"User: {user_input} | AI: {response}"
        pending_conversations.append(conversation_entry)

        if len(pending_conversations) >= CONVERSATION_BATCH_SIZE:
            logger.info("Batch size reached. Summarizing and storing...")
            summarization_prompt = f"""Summarize the following conversation batch:
{chr(10).join(pending_conversations)}
Respond with ONLY a JSON object:
{{"summary": "concise summary"}}"""
            try:
                summary_response = llm.invoke([SystemMessage(content=summarization_prompt)]).content.strip()
                summary_data = extract_json_response(summary_response, {"summary": "Summary of conversation batch."})
                summary_text = summary_data.get("summary", "Summary of conversation batch.")
            except Exception as e:
                logger.error(f"Summarization failed: {e}")
                summary_text = "Summary of conversation batch."

            doc_to_store = {
                "user_id": user_id,
                "data": {},
                "conversation_history": pending_conversations,
                "summary": summary_text,
                "category": "conversation_batch"
            }
            try:
                vectorstore.add_texts(
                    texts=[json.dumps(doc_to_store)],
                    metadatas=[{"user_id": user_id, "category": "conversation_batch", "timestamp": datetime.now().isoformat()}]
                )
                logger.info("Stored summarized conversation batch.")
            except Exception as e:
                logger.error(f"Failed to store batch summary in Chroma: {e}")
            pending_conversations = []

        storage_prompt = f"""Analyze this conversation and decide what profile information to store.
User Input: "{user_input}"
AI Response: "{response}"
Current Tone: {tone}
Respond with ONLY a JSON object:
{{"store": true/false, "data": {{}}, "category": "category_name", "summary": "brief summary"}}"""
        try:
            storage_response = llm.invoke([SystemMessage(content=storage_prompt)]).content.strip()
            storage_data = extract_json_response(storage_response, {
                "store": True,
                "data": {"last_interaction": user_input[:100]},
                "category": "conversation",
                "summary": f"Conversation about: {user_input[:50]}..."
            })
            if storage_data.get("store", False):
                new_data = storage_data.get("data", {})
                user_profile.setdefault("data", {}).update(new_data)
                user_profile.setdefault("conversation_history", []).append(conversation_entry)
                user_profile["conversation_history"] = user_profile["conversation_history"][-10:]
                doc_to_store = {
                    "user_id": user_id,
                    "data": new_data,
                    "conversation_history": [conversation_entry],
                    "summary": storage_data.get("summary", ""),
                    "category": storage_data.get("category", "general")
                }
                try:
                    vectorstore.add_texts(
                        texts=[json.dumps(doc_to_store)],
                        metadatas=[{"user_id": user_id, "category": storage_data.get("category", "general"), "timestamp": datetime.now().isoformat()}]
                    )
                    logger.info("Stored individual conversation data.")
                except Exception as e:
                    logger.error(f"Failed to store conversation: {e}")
        except Exception as e:
            logger.error(f"Storage decision failed: {e}")

        state.update({"user_profile": user_profile, "conversation_history": user_profile["conversation_history"], "pending_conversations": pending_conversations})
        return state

    else:
        logger.info("Memory Agent: No action needed, passing through")
        return state
