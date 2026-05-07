import streamlit as st
import os

from agent.config import AgentConfig, validate_key
from agent.executor import build_executor
from agent.intents import classify_intent
from agent.state import init_state, SessionManager
from agent.tools import build_tools

from langchain_community.chat_models.tongyi import ChatTongyi

st.set_page_config(page_title="法智AI - 法律助手", page_icon="⚖️", layout="wide")
st.title("⚖️ 法智AI — 法律助手智能体")
st.caption("智能案件分类 | 法律风险评估 | 文书模板生成 | 法典RAG检索 | 联网搜索")

cfg = AgentConfig()
init_state()

try:
    validate_key(cfg)
except Exception as e:
    st.error(f"配置错误：{str(e)}")
    st.stop()


@st.cache_resource(ttl=3600)
def get_executor():
    llm = ChatTongyi(model=cfg.llm_model, temperature=0.1)
    tools = build_tools()
    executor = build_executor(llm, tools, cfg)
    return llm, executor


llm, executor = get_executor()

with st.sidebar:
    st.subheader("🖥 服务状态")
    st.success("✅ LLM 模型: qwen3-max")
    st.success(f"✅ 百度搜索: {'已配置' if cfg.baidu_api_key else '未配置'}")
    st.success("✅ Embedding API 已配置")

    st.divider()
    st.subheader("会话管理")
    if st.button("🔄 清空会话"):
        SessionManager.clear_session()
        st.rerun()

    with st.expander("📊 会话统计"):
        summary = SessionManager.get_session_summary()
        st.markdown(f"消息数量: {summary['message_count']}")
        st.markdown(f"开始时间: {summary['start_time'][:19]}")
        usage = summary.get('tool_usage', {})
        if usage:
            st.markdown("**工具使用**:")
            for tool, count in usage.items():
                st.markdown(f"- {tool}: {count}次")

    with st.expander("💡 示例咨询"):
        st.markdown("**📋 案件分类**")
        st.markdown("- 朋友借我10万不还，这种情况属于什么案件？")
        st.markdown("- 我被邻居打了，该去哪起诉？")
        st.markdown("**⚠️ 风险评估**")
        st.markdown("- 我和合伙人只有口头协议，有什么风险？")
        st.markdown("- 签了格式合同里违约金写得很高，需要担心吗？")
        st.markdown("**📝 文书模板**")
        st.markdown("- 帮我生成一份离婚协议书模板")
        st.markdown("- 我需要一份劳动合同的模板")
        st.markdown("**📚 法规检索**")
        st.markdown("- 民法典中关于合同违约金的规定")
        st.markdown("- 故意伤害罪的量刑标准是什么？")

for msg in st.session_state["agent_messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("请输入法律问题，例如：民法典中关于借款合同利息的规定是什么？")
if query:
    SessionManager.add_to_memory({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("⚖️ 法智AI 分析中..."):
            try:
                intent = classify_intent(llm, query)
                history_text = SessionManager.get_recent_context()
                result = executor.invoke({"input": query, "chat_history": history_text})

                answer = result.get("output", "未生成结果")
                steps = result.get("intermediate_steps", [])
                status = result.get("status", "success")

                emoji = "✅" if status == "success" else "❌"
                formatted = f"### 任务分类\n- intent: `{intent.intent}`\n- reason: {intent.reason}\n\n### 执行结果\n{emoji} {answer}"
                st.markdown(formatted)

                with st.expander("🔍 查看推理过程"):
                    if not steps:
                        st.info("此任务直接由工具处理，未经过Agent多步推理。")
                    else:
                        for i, step in enumerate(steps, start=1):
                            st.markdown(f"**Step {i}**")
                            if step.get("type") == "tool_call":
                                tool_name = step.get("tool", "unknown")
                                st.markdown(f"- ⚡ 调用工具: `{tool_name}`")
                                st.markdown(f"- 📥 输入参数: {step.get('input', {})}")
                                SessionManager.update_tool_usage(tool_name)
                            elif step.get("type") == "tool_response":
                                output_text = str(step.get("output", "无响应"))
                                if len(output_text) > 500:
                                    st.markdown(f"- 📤 返回结果: {output_text[:500]}...")
                                else:
                                    st.markdown(f"- 📤 返回结果: {output_text}")
                            st.markdown("---")

                SessionManager.add_to_memory({"role": "assistant", "content": formatted})
            except Exception as e:
                err = f"❌ 执行失败：{str(e)}"
                st.error(err)
                SessionManager.add_to_memory({"role": "assistant", "content": err})
