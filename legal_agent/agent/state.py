import streamlit as st
from datetime import datetime


class SessionManager:
    @staticmethod
    def init_state():
        st.session_state.setdefault("agent_messages", [])
        st.session_state.setdefault("conversation_history", [])
        st.session_state.setdefault("tool_usage", {})
        st.session_state.setdefault("session_metadata", {
            "start_time": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
            "message_count": 0,
        })

    @staticmethod
    def update_last_activity():
        st.session_state["session_metadata"]["last_activity"] = datetime.now().isoformat()

    @staticmethod
    def increment_message_count():
        st.session_state["session_metadata"]["message_count"] += 1

    @staticmethod
    def add_to_memory(message: dict):
        st.session_state["agent_messages"].append(message)
        st.session_state["conversation_history"].append({
            "timestamp": datetime.now().isoformat(),
            "message": message,
        })
        if len(st.session_state["conversation_history"]) > 100:
            st.session_state["conversation_history"] = st.session_state["conversation_history"][-100:]
        SessionManager.update_last_activity()
        SessionManager.increment_message_count()

    @staticmethod
    def update_tool_usage(tool_name: str):
        if tool_name:
            st.session_state["tool_usage"][tool_name] = st.session_state["tool_usage"].get(tool_name, 0) + 1

    @staticmethod
    def get_recent_context(max_messages: int = 10) -> str:
        recent = st.session_state["agent_messages"][-max_messages:]
        return "\n".join([f"{m['role']}: {m['content'][:200]}" for m in recent])

    @staticmethod
    def get_session_summary() -> dict:
        meta = st.session_state["session_metadata"]
        return {
            "message_count": meta["message_count"],
            "start_time": meta["start_time"],
            "last_activity": meta["last_activity"],
            "tool_usage": dict(st.session_state["tool_usage"]),
        }

    @staticmethod
    def clear_session():
        st.session_state["agent_messages"] = []
        st.session_state["tool_usage"] = {}
        st.session_state["session_metadata"]["message_count"] = 0
        st.session_state["session_metadata"]["last_activity"] = datetime.now().isoformat()


def init_state():
    SessionManager.init_state()
