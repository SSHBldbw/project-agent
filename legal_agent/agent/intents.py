from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi


class IntentResult(BaseModel):
    intent: str = Field(description="任务类别: law_rag/case_classify/risk_assess/doc_template/general")
    reason: str = Field(description="判定理由")


def classify_intent(llm: ChatTongyi, user_query: str) -> IntentResult:
    parser = JsonOutputParser(pydantic_object=IntentResult)
    prompt = PromptTemplate(
        template=(
            "你是法律任务分类器，请判断用户请求最适合的任务类别。\n"
            "候选类别:\n"
            "- case_classify: 分析案件类型、管辖、诉讼时效等\n"
            "- risk_assess: 评估法律风险、识别风险点\n"
            "- doc_template: 需要法律文书模板（起诉状、离婚协议、合同等）\n"
            "- law_rag: 查询法典法律条文、法律基础知识（民法典/刑法典等）\n"
            "- general: 通用法律咨询、其他法律问题\n"
            "{format_instructions}\n"
            "用户请求: {query}"
        ),
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm | parser
    data = chain.invoke({"query": user_query})
    return IntentResult(**data)
