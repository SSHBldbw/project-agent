import time
import logging
from typing import Dict, List, Any

from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.tools import StructuredTool

from agent.config import AgentConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AgentExecutor")


class EnhancedAgentExecutor:
    def __init__(self, llm: ChatTongyi, tools: List[StructuredTool], config: AgentConfig):
        self.llm = llm
        self.tools = tools
        self.config = config

        system_prompt = (
            "你是专业的法律助理智能体——「法智AI」。你拥有中国法律法规知识库，能辅助用户进行法律咨询。\n\n"
            "核心能力：\n"
            "1. 案件分类：使用「case_classify」智能识别案件类型（民事/刑事/行政）、案由、管辖法院和诉讼时效\n"
            "2. 风险评估：使用「risk_assess」识别法律场景中的潜在风险点、风险等级和潜在后果\n"
            "3. 文书模板：使用「doc_template」生成起诉状、离婚协议、借款合同等法律文书标准模板\n"
            "4. 法规检索：使用「law_rag」基于法典全文进行语义检索（民法典、刑法典等）\n"
            "5. 联网搜索：使用「search」查询最新法律法规动态和司法解释\n\n"
            "工作规范：\n"
            "- 法律条文查询优先使用「law_rag」\n"
            "- 需要最新信息时使用「search」联网搜索\n"
            "- 案情分析时先用「case_classify」确定案件性质\n"
            "- 合同/交易/纠纷类问题可先用「risk_assess」评估风险\n"
            "- 用户需要起草法律文书时使用「doc_template」\n"
            "- 始终遵循事实依据，不编造法律条文\n"
            "- 每次回答末尾添加「⚠️ 以上信息仅供参考，不构成法律意见，请咨询专业律师。」\n\n"
            "语言风格：专业严谨，分点作答，先结论后依据。"
        )

        self.agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
        )

    def _retry_with_backoff(self, func, max_retries: int = 3, base_delay: float = 1.0):
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                logger.info(f"执行尝试 {attempt + 1}/{max_retries + 1}")
                return func()
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"执行失败 (尝试 {attempt + 1}): {str(e)}. 等待 {delay:.2f}秒后重试...")
                    time.sleep(delay)
                else:
                    logger.error(f"执行失败，已达到最大重试次数: {str(e)}")
        raise RuntimeError(f"工具执行失败，已重试 {max_retries} 次: {str(last_error)}")

    def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            user_input = input_data.get("input", "")
            chat_history = input_data.get("chat_history", "")

            logger.info(f"接收到法律任务: {user_input}")

            full_input = user_input
            if chat_history:
                full_input = f"对话历史:\n{chat_history}\n\n当前问题: {user_input}"

            result = self._retry_with_backoff(
                lambda: self.agent.invoke({"messages": [{"role": "user", "content": full_input}]}),
                max_retries=self.config.max_retries,
                base_delay=self.config.retry_delay,
            )

            messages = result.get("messages", [])
            output = ""
            intermediate_steps = []

            for msg in messages:
                content = getattr(msg, "content", "")
                tool_calls = getattr(msg, "tool_calls", None)

                if tool_calls:
                    for tc in tool_calls:
                        intermediate_steps.append({
                            "type": "tool_call",
                            "tool": tc.get("name", "unknown"),
                            "input": tc.get("args", {}),
                        })
                elif content and getattr(msg, "type", "") == "tool":
                    intermediate_steps.append({
                        "type": "tool_response",
                        "tool": getattr(msg, "name", "unknown"),
                        "output": content,
                    })
                elif content and getattr(msg, "type", "") in ("ai", "assistant"):
                    intermediate_steps.append({
                        "type": "tool_response",
                        "tool": "assistant",
                        "output": content,
                    })

            if messages:
                last_msg = messages[-1]
                if hasattr(last_msg, "content") and last_msg.content:
                    output = last_msg.content

            logger.info("Agent执行成功")
            return {
                "output": output or "任务已完成",
                "intermediate_steps": intermediate_steps,
                "status": "success",
            }
        except Exception as e:
            logger.error(f"Agent执行失败: {str(e)}")
            return {
                "output": str(e),
                "intermediate_steps": [],
                "status": "error",
            }


def build_executor(llm: ChatTongyi, tools: List[StructuredTool], config: AgentConfig = None) -> EnhancedAgentExecutor:
    if config is None:
        config = AgentConfig()
    return EnhancedAgentExecutor(llm, tools, config)
