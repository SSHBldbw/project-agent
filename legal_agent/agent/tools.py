from __future__ import annotations

import json
import math
import time
import os
import re
import requests
from typing import List, Tuple

from langchain_core.tools import StructuredTool
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from agent.config import AgentConfig

# ---- 向量库 + BM25 全文 chunk 缓存 ----
_VECTORSTORE = None
_VS_CHUNKS: List[Tuple[str, str]] = []

# ---- 重排序模型 (CrossEncoder) ----
_RERANKER = None
_RERANKER_PATH = r"D:\AI_\test\advanced_rag\models_baaI\BAAI\bge-reranker-base"


def _get_reranker():
    global _RERANKER
    if _RERANKER is not None:
        return _RERANKER
    from sentence_transformers import CrossEncoder
    _RERANKER = CrossEncoder(_RERANKER_PATH)
    return _RERANKER


def _read_file_auto_encoding(file_path: str) -> str:
    for enc in ("gbk", "gb18030", "utf-8", "gb2312", "big5"):
        try:
            with open(file_path, "r", encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    with open(file_path, "r", encoding="gbk", errors="replace") as f:
        return f.read()


# ======================== 案件类型智能分类 ========================

_CASE_PATTERNS = [
    (r"打人|伤害|伤人|重伤|轻伤|故意杀人|过失致死|杀人|抢劫|抢|盗窃|偷|诈骗|骗|强奸|猥亵|寻衅滋事|聚众斗殴|涉黑|贩毒|吸毒|贪污|受贿|挪用公款|渎职", "刑事"),
    (r"合同|借|欠|债务|违约|赔偿|离婚|继承|抚养|赡养|房产|物业|劳动|干活|受伤|工伤|工资|加班|社保|消费|侵权|撞|损害|名誉|隐私|知识产权|专利|商标|著作权|公司|股权", "民事"),
    (r"行政|处罚|罚款|吊销|许可|复议|拆迁|征收|信息公开|行政诉讼|国家赔偿|交警|城管", "行政"),
]

_CASE_SUBTYPES = {
    "刑事": {
        r"盗窃|偷|盗": ("盗窃罪", "刑法第264条", "基层人民法院", "公诉机关举证"),
        r"诈骗|骗": ("诈骗罪", "刑法第266条", "基层人民法院", "公诉机关举证"),
        r"伤害|打人|伤人|重伤|轻伤": ("故意伤害罪", "刑法第234条", "基层人民法院", "公诉机关举证"),
        r"抢劫|抢": ("抢劫罪", "刑法第263条", "中级人民法院", "公诉机关举证"),
        r"贪污|受贿|挪用": ("贪污贿赂罪", "刑法第382-396条", "中级人民法院", "公诉机关举证"),
        r"交通肇事|危险驾驶|醉驾": ("交通肇事/危险驾驶罪", "刑法第133条", "基层人民法院", "公诉机关举证"),
    },
    "民事": {
        r"合同|违约": ("合同纠纷", "民法典合同编", "被告住所地/合同履行地法院", "谁主张谁举证"),
        r"借|欠|债务": ("民间借贷/债务纠纷", "民法典合同编", "被告住所地/合同履行地法院", "谁主张谁举证"),
        r"离婚": ("离婚纠纷", "民法典婚姻家庭编", "被告住所地法院", "谁主张谁举证"),
        r"继承|遗嘱|遗产": ("继承纠纷", "民法典继承编", "被继承人死亡时住所地/主要遗产所在地法院", "谁主张谁举证"),
        r"劳动|干活|受伤|工伤|工资|加班|社保": ("劳动争议/人身损害", "劳动法/劳动合同法/民法典侵权编", "用人单位所在地/劳动合同履行地/侵权行为地", "用人单位/侵权人承担主要举证责任"),
        r"侵权|撞|损害|赔偿": ("侵权责任纠纷", "民法典侵权责任编", "侵权行为地/被告住所地法院", "谁主张谁举证"),
        r"房产|物业|房屋": ("不动产纠纷", "民法典物权编", "不动产所在地法院（专属管辖）", "谁主张谁举证"),
        r"知识产权|专利|商标|著作权": ("知识产权纠纷", "专利法/商标法/著作权法", "中级人民法院（一般）", "谁主张谁举证"),
        r"公司|股权|股东": ("公司纠纷", "公司法", "公司住所地法院", "谁主张谁举证"),
        r"消费|消费者|假货|退款|退货": ("消费纠纷", "消费者权益保护法", "被告住所地/侵权行为地法院", "谁主张谁举证"),
    },
    "行政": {
        r"处罚|罚款|吊销|交警|城管": ("行政处罚纠纷", "行政处罚法", "被告行政机关所在地法院", "被告行政机关举证"),
        r"征收|拆迁|征用": ("征收补偿纠纷", "土地管理法/国有土地上房屋征收与补偿条例", "不动产所在地法院", "被告行政机关举证"),
        r"信息公开": ("政府信息公开纠纷", "政府信息公开条例", "被告行政机关所在地法院", "被告行政机关举证"),
    },
}

_TIMELINE = {
    "刑事": "追诉时效根据最高刑期而定：不满5年→5年，5-10年→10年，10年以上→15年，死刑/无期→20年",
    "民事": "普通诉讼时效3年，自权利人知道权利受损及义务人之日起算",
    "行政": "复议申请60日内，行政诉讼6个月内，自知道行政行为之日起算",
}


class CaseClassifyInput(BaseModel):
    description: str = Field(description="案件事实描述，越详细分析越准确")


def case_classify_tool(description: str) -> str:
    text = description[:2000]

    case_type = "民事"
    for pattern, ctype in _CASE_PATTERNS:
        if re.search(pattern, text):
            case_type = ctype
            break

    subtype_info = ("其他", "请进一步明确案情", "请进一步明确", "请进一步明确")
    subtype_map = _CASE_SUBTYPES.get(case_type, {})
    for pattern, info in subtype_map.items():
        if re.search(pattern, text):
            subtype_info = info
            break

    timeline = _TIMELINE.get(case_type, "")

    lines = [
        "## 案件类型分析\n",
        f"**案件类型**: {case_type}案件",
        f"**案由分类**: {subtype_info[0]}",
        f"**适用法律**: {subtype_info[1]}",
        f"**管辖法院**: {subtype_info[2]}",
        f"**举证责任**: {subtype_info[3]}",
        f"**诉讼时效**: {timeline}",
        "",
        "### 下一步建议",
        f"1. 收集整理相关证据材料（合同、转账记录、聊天记录、证人证言等）",
        f"2. 确认{case_type}诉讼时效，避免过期",
        f"3. 如需起诉，向{'-'.join(subtype_info[2].split('、')[:1])}提交起诉状",
        "4. 复杂案件建议委托专业律师代理",
    ]
    return "\n".join(lines)


# ======================== 法律风险评估 ========================

_RISK_PATTERNS = {
    r"违约金.{0,5}(高|过|多)": ("合同违约金过高风险", "高", "可能被法院调低，实际赔付可能远低于约定"),
    r"定金.{0,5}(不退|没收)": ("定金罚则风险", "高", "给付方违约将丧失定金，收受方违约需双倍返还"),
    r"担保|保证人|连带保证": ("担保责任风险", "高", "连带保证人需以个人全部财产承担清偿责任"),
    r"没有.{0,3}合同|口头|未签": ("无书面合同风险", "高", "口头协议举证困难，发生纠纷时维权成本极高"),
    r"借条|欠条.{0,5}(没有|未写|未签)": ("债权凭证缺失风险", "高", "缺乏书面凭证可能导致债权难以追索"),
    r"诉讼时效|过期.{0,3}年": ("诉讼时效风险", "高", "超过诉讼时效将丧失胜诉权，无法通过诉讼维权"),
    r"知识产权|专利|商标.{0,5}(侵权|盗用|抄袭)": ("知识产权侵权风险", "中", "可能面临停止侵权、赔偿损失、行政处罚等后果"),
    r"婚前.{0,3}财产|个人财产": ("婚前财产混同风险", "中", "婚前财产与婚后财产混同后可能被认定为夫妻共同财产"),
    r"竞业限制|商业秘密|保密": ("商业秘密/竞业限制风险", "中", "违反保密义务或竞业限制需承担赔偿及违约责任"),
    r"格式合同|格式条款|霸王条款": ("格式条款无效风险", "中", "不合理的格式条款可能被认定无效"),
    r"继承|遗嘱.{0,5}(没写|未立)": ("无遗嘱继承风险", "中", "按法定继承可能不符合被继承人真实意愿"),
    r"工伤|事故|安全": ("人身安全/工伤风险", "高", "可能面临民事赔偿、行政处罚甚至刑事责任"),
}

_RISK_LEVEL_DESC = {
    "高": "[高风险] 可能导致重大经济损失或法律后果，建议立即采取应对措施",
    "中": "[中风险] 存在潜在法律隐患，建议尽快完善相关手续",
    "低": "[低风险] 风险可控，但仍需保持关注",
}


class RiskAssessInput(BaseModel):
    scenario: str = Field(description="待评估的法律场景描述")


def risk_assess_tool(scenario: str) -> str:
    text = scenario[:2000]
    findings = []

    for pattern, (name, level, consequence) in _RISK_PATTERNS.items():
        if re.search(pattern, text):
            findings.append((name, level, consequence))

    if not findings:
        high_count = len(re.findall(r"(合同|违约|赔偿|侵权|诉讼|仲裁|判决)", text))
        if high_count >= 2:
            findings.append(("综合法律风险", "中", "存在多个法律要素，建议全面审查"))
        else:
            findings.append(("暂无显著风险", "低", "未发现明显法律风险点，但仍建议定期审查"))

    lines = ["## 法律风险评估报告\n"]
    lines.append(f"评估场景: {text[:120]}...")
    lines.append(f"检测到 {len(findings)} 个风险点:\n")

    for i, (name, level, consequence) in enumerate(findings, 1):
        lines.append(f"### 风险 {i}: {name}")
        lines.append(f"风险等级: {_RISK_LEVEL_DESC.get(level, level)}")
        lines.append(f"潜在后果: {consequence}")
        lines.append("")

    lines.append("### 综合建议")
    high_risks = [f for f in findings if f[1] == "高"]
    mid_risks = [f for f in findings if f[1] == "中"]
    if high_risks:
        lines.append(f"⚠️ 发现 {len(high_risks)} 项高风险，强烈建议咨询专业律师评估具体法律风险")
    if mid_risks:
        lines.append(f"⚡ 发现 {len(mid_risks)} 项中风险，建议尽快完善相关法律文件和手续")
    if not high_risks and not mid_risks:
        lines.append("当前风险可控，建议保持合规意识，定期审查")

    return "\n".join(lines)


# ======================== 法律文书模板生成 ========================

_DOC_TEMPLATES = {
    "起诉状|民事起诉状|民事诉讼": {
        "title": "民事起诉状",
        "structure": [
            "一、原告信息（姓名、性别、出生日期、身份证号、住址、联系方式）",
            "二、被告信息（同上）",
            "三、诉讼请求（明确具体的诉求，如：请求判令被告支付XX元）",
            "四、事实与理由（按时间顺序陈述事实，引用法律依据）",
            "五、证据清单及证据来源（列举所有证据并说明证明目的）",
            "六、此致 XX人民法院",
            "七、具状人签名及日期",
        ],
        "tips": [
            "诉讼请求必须明确具体，不能模糊",
            "事实描述要有时间、地点、人物、经过四要素",
            "相关法条建议先用 law_rag 检索确认",
            "证据材料需提交复印件，原件开庭时出示",
        ],
    },
    "离婚协议|离婚协议书|离婚": {
        "title": "离婚协议书",
        "structure": [
            "一、男女双方基本信息（姓名、身份证号、结婚日期）",
            "二、自愿离婚的意思表示",
            "三、子女抚养安排（抚养权归属、抚养费金额及支付方式、探视权约定）",
            "四、财产分割（房产、车辆、存款、股票等逐一列明归属）",
            "五、债务处理（夫妻共同债务及个人债务的承担方式）",
            "六、其他约定（如离婚后不得干扰对方生活等）",
            "七、双方签名及日期",
        ],
        "tips": [
            "财产分割需逐一列明，避免笼统概括",
            "抚养费建议约定每年递增比例",
            "大额债务需双方签字确认",
            "建议公证或经法院确认以增强执行力",
        ],
    },
    "借款合同|借条|借款协议": {
        "title": "借款合同/借条",
        "structure": [
            "一、出借人信息（姓名、身份证号、联系方式）",
            "二、借款人信息（姓名、身份证号、联系方式）",
            "三、借款金额（大写+小写，如：人民币壹拾万元整 ¥100,000）",
            "四、借款利率（年利率/月利率，不得超过LPR的4倍）",
            "五、借款期限（起止日期）",
            "六、还款方式（一次性还本付息/分期还本付息）",
            "七、担保条款（如有担保人或抵押物）",
            "八、违约责任约定",
            "九、争议解决方式（诉讼/仲裁）",
            "十、双方签名、按手印及日期",
        ],
        "tips": [
            "大额借款建议通过银行转账留存凭证",
            "利率不得违反国家规定上限",
            "务必双方签名并按手印",
            "建议拍照或录像留存签署过程",
        ],
    },
    "劳动合同|劳务合同|聘用": {
        "title": "劳动合同",
        "structure": [
            "一、用人单位信息（名称、住所、法定代表人）",
            "二、劳动者信息（姓名、身份证号、住址）",
            "三、合同期限（固定期限/无固定期限/以完成一定工作任务为期限）",
            "四、工作内容与工作地点",
            "五、工作时间与休息休假",
            "六、劳动报酬（工资、奖金、津贴、加班费计算方式）",
            "七、社会保险与福利待遇",
            "八、劳动保护与劳动条件",
            "九、合同变更、解除与终止条件",
            "十、违约责任与争议解决",
            "十一、双方签名盖章及日期",
        ],
        "tips": [
            "必须包含《劳动合同法》第十七条规定的必备条款",
            "试用期不得超过法定期限",
            "工资不得低于当地最低工资标准",
            "建议参考当地人社部门示范文本",
        ],
    },
    "租赁合同|租房|房屋租赁": {
        "title": "房屋租赁合同",
        "structure": [
            "一、出租方信息（姓名/名称、身份证号、联系方式）",
            "二、承租方信息（同上）",
            "三、房屋基本信息（坐落位置、面积、房产证号）",
            "四、租赁期限（起止日期）",
            "五、租金及支付方式（金额、支付周期）",
            "六、押金条款（金额、退还条件）",
            "七、房屋使用与维护责任",
            "八、转租与续租约定",
            "九、合同解除条件",
            "十、违约责任",
            "十一、双方签名及日期",
        ],
        "tips": [
            "签约前核实房产证和出租方身份",
            "明确水电燃气物业费等承担方",
            "押金退还条件应明确具体",
            "租赁合同最长不得超过20年",
        ],
    },
    "授权委托书|委托书|委托代理": {
        "title": "授权委托书",
        "structure": [
            "一、委托人信息（姓名、身份证号、联系方式）",
            "二、受托人信息（姓名、身份证号、联系方式）",
            "三、委托事项（明确具体的代理事项）",
            "四、授权范围（一般授权/特别授权，特别授权需列明权限）",
            "五、委托期限",
            "六、转委托权限（是否允许转委托）",
            "七、委托人签名按手印及日期",
        ],
        "tips": [
            "特别授权需明确列出具体权限",
            "涉及重大事项建议公证",
            "注明委托人及受托人身份证复印件附后",
        ],
    },
}


class DocTemplateInput(BaseModel):
    doc_type: str = Field(description="需要的文书类型，如：起诉状、离婚协议、借款合同、劳动合同等")


def doc_template_tool(doc_type: str) -> str:
    matched_template = None
    for keywords, template in _DOC_TEMPLATES.items():
        if re.search(keywords, doc_type):
            matched_template = template
            break

    if not matched_template:
        available = "、".join(t["title"] for t in _DOC_TEMPLATES.values())
        return f"未找到匹配的文书模板。目前已支持的模板类型: {available}\n\n请重新描述您需要的文书类型。"

    lines = [
        f"## {matched_template['title']} — 标准模板\n",
        "### 📋 基本结构\n",
    ]
    for item in matched_template["structure"]:
        lines.append(f"- {item}")

    lines.append("\n### 💡 注意事项\n")
    for tip in matched_template["tips"]:
        lines.append(f"- {tip}")

    lines.append(f"\n---\n⚠️ 以上为通用模板，具体条款建议结合实际情况调整，必要时委托律师审核。")

    return "\n".join(lines)


# ======================== 百度 MCP 搜索 ========================

class SearchInput(BaseModel):
    keyword: str = Field(description="搜索查询关键词或短语")


def _mcp_call_tool(tool_name: str, arguments: dict, max_retries: int = 2) -> dict:
    config = AgentConfig()
    baidu_api_key = config.baidu_api_key
    if not baidu_api_key:
        return {"ok": False, "error": "BAIDU_API_KEY 未配置", "results": []}

    mcp_url = "https://qianfan.baidubce.com/v2/tools/web-search/mcp"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {baidu_api_key}"}

    import uuid

    for attempt in range(max_retries + 1):
        try:
            _payload = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
                "id": str(uuid.uuid4()),
            }
            resp = requests.post(mcp_url, headers=headers, json=_payload, timeout=25)

            if resp.status_code == 200:
                data = resp.json()
                content = data.get("result", {}).get("content", [])
                results = []
                for item in content:
                    text = item.get("text", "")
                    if text:
                        results.append(text)
                return {"ok": True, "provider": "baidu_qianfan_mcp", "results": results}

            if attempt < max_retries:
                time.sleep(1)
        except Exception as e:
            if attempt < max_retries:
                time.sleep(1)
            else:
                return {"ok": False, "error": f"MCP 调用异常: {str(e)}", "results": []}

    return {"ok": False, "error": "MCP 调用失败", "results": []}


def search_tool(keyword: str) -> str:
    result = _mcp_call_tool("webSearch", {"query": keyword, "count": 3})
    if result.get("ok"):
        normalized = []
        for block in result.get("results", []):
            parts = _parse_mcp_web_block(block)
            normalized.append(parts)
        return json.dumps({"ok": True, "provider": "baidu_qianfan_mcp", "query": keyword, "results": normalized[:5]}, ensure_ascii=False, indent=2)
    return json.dumps({"ok": False, "provider": "baidu_qianfan_mcp", "query": keyword, "error": result.get("error", "未知错误"), "results": []}, ensure_ascii=False)


def _parse_mcp_web_block(block: str) -> dict:
    title = ""
    content = ""
    url = ""
    for line in block.split("\n"):
        if line.startswith("Title:"):
            title = line[6:].strip()
        elif line.startswith("Content:"):
            content = line[8:].strip()
        elif line.startswith("URL:"):
            url = line[4:].strip()
    return {"title": title[:100], "snippet": content[:400], "url": url}


# ======================== 法典混合检索 ========================

class LawRAGInput(BaseModel):
    question: str = Field(description="法律咨询问题，基于多法典混合检索")


def _init_vectorstore(force_rebuild: bool = False):
    global _VECTORSTORE, _VS_CHUNKS
    if _VECTORSTORE is not None and not force_rebuild:
        return _VECTORSTORE

    config = AgentConfig()

    from langchain_openai import OpenAIEmbeddings
    from langchain_chroma import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    base_dir = os.path.dirname(os.path.dirname(__file__))

    law_files = sorted(
        [os.path.join(base_dir, f) for f in os.listdir(base_dir)
         if f.endswith(".txt") and "中华" in f],
    )
    if not law_files:
        raise FileNotFoundError('未找到法典文件（含"中华"的 .txt 文件）')

    expected_labels = {
        os.path.basename(fp).replace("中华人民共和国", "").replace(".txt", "")[:12]
        for fp in law_files
    }

    embeddings = OpenAIEmbeddings(
        model=config.embedding_model,
        openai_api_key=config.embedding_api_key,
        openai_api_base=config.embedding_base_url,
        chunk_size=30,
    )

    persist_dir = os.path.join(os.getenv("LAW_CHROMA_PERSIST_DIR", ".chroma_legal"), "law_codes")
    os.makedirs(persist_dir, exist_ok=True)

    need_rebuild = force_rebuild

    if not need_rebuild:
        try:
            _VECTORSTORE = Chroma(
                embedding_function=embeddings,
                persist_directory=persist_dir,
                collection_name="law_codes",
            )
            count = _VECTORSTORE._collection.count()
            if count > 0:
                data = _VECTORSTORE._collection.get(include=["documents", "metadatas"])
                embedded_labels = set()
                for text, meta in zip(data.get("documents", []), data.get("metadatas", [])):
                    src = (meta or {}).get("source", "")
                    embedded_labels.add(src)
                    _VS_CHUNKS.append((text, src))

                missing = expected_labels - embedded_labels
                if missing:
                    print(f"[向量库] 警告: 缺少法典 {missing}，将重建完整索引")
                    need_rebuild = True
                    _VECTORSTORE = None
                    _VS_CHUNKS.clear()
                else:
                    print(f"[向量库] 加载已有索引: {count} 条, 法典: {embedded_labels}")
                    return _VECTORSTORE
            else:
                need_rebuild = True
        except Exception as e:
            print(f"[向量库] 加载失败({e})，将重建")
            import shutil
            shutil.rmtree(persist_dir, ignore_errors=True)
            os.makedirs(persist_dir, exist_ok=True)
            need_rebuild = True
            _VECTORSTORE = None

    if need_rebuild:
        import shutil
        shutil.rmtree(persist_dir, ignore_errors=True)
        os.makedirs(persist_dir, exist_ok=True)
        _VECTORSTORE = None
        _VS_CHUNKS.clear()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=60,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )

    all_docs: List[Document] = []
    for file_path in law_files:
        label = os.path.basename(file_path).replace("中华人民共和国", "").replace(".txt", "")[:12]
        text = _read_file_auto_encoding(file_path)
        if not text:
            raise IOError(f"无法读取法典文件: {file_path}")
        chunks = splitter.split_text(text)
        print(f"[向量库] {label}: {len(chunks)} 条待嵌入")
        for c in chunks:
            all_docs.append(Document(page_content=c, metadata={"source": label}))

    _VS_CHUNKS = [(d.page_content, d.metadata.get("source", "")) for d in all_docs]

    BATCH_SIZE = 20
    MAX_RETRY_PER_BATCH = 3
    total = len(all_docs)
    _VECTORSTORE = None
    _failed = 0

    for i in range(0, total, BATCH_SIZE):
        batch = all_docs[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = math.ceil(total / BATCH_SIZE)

        success = False
        for retry in range(MAX_RETRY_PER_BATCH):
            try:
                if _VECTORSTORE is None:
                    _VECTORSTORE = Chroma.from_documents(
                        documents=batch,
                        embedding=embeddings,
                        persist_directory=persist_dir,
                        collection_name="law_codes",
                    )
                else:
                    _VECTORSTORE.add_documents(batch)
                success = True
                break
            except Exception as e:
                wait = 1.0 * (retry + 1)
                print(f"[向量库] 批次 {batch_num}/{total_batches} 第{retry+1}次失败: {e}，{wait}s后重试...")
                time.sleep(wait)

        if success:
            print(f"[向量库] 批次 {batch_num}/{total_batches} ({len(batch)}条) OK")
            time.sleep(0.3)
        else:
            _failed += len(batch)
            print(f"[向量库] 批次 {batch_num}/{total_batches} FAIL")

    print(f"[向量库] 总计 {total} 条, 成功 {total - _failed}, 失败 {_failed}")
    return _VECTORSTORE


def _bm25_search(query: str, top_k: int = 20) -> List[Tuple[int, float]]:
    global _VS_CHUNKS
    if not _VS_CHUNKS:
        return []

    from rank_bm25 import BM25Okapi
    import jieba

    _tokenized = [list(jieba.cut(t[0])) for t in _VS_CHUNKS]
    bm25 = BM25Okapi(_tokenized)
    q_tokens = list(jieba.cut(query))
    scores = bm25.get_scores(q_tokens)

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [(idx, float(s)) for idx, s in ranked if s > 0]


def _hybrid_rerank(question: str, top_k: int = 5, vec_k: int = 20, bm25_k: int = 20) -> List[Tuple[Document, float]]:
    global _VS_CHUNKS

    vectorstore = _init_vectorstore()

    short = question[:120]
    vec_results = vectorstore.similarity_search_with_relevance_scores(short, k=vec_k)

    bm25_candidates = _bm25_search(question, top_k=bm25_k)

    seen = {}
    candidates = []

    for doc, score in vec_results:
        key = doc.page_content[:80]
        if key not in seen:
            seen[key] = len(candidates)
            candidates.append((doc, score))

    for idx, bm_score in bm25_candidates:
        text = _VS_CHUNKS[idx][0] if idx < len(_VS_CHUNKS) else ""
        if not text:
            continue
        key = text[:80]
        if key not in seen:
            doc = Document(page_content=text, metadata={"source": _VS_CHUNKS[idx][1]})
            candidates.append((doc, bm_score / max(1, max(s for _, s in bm25_candidates))))

    if not candidates:
        return []

    try:
        reranker = _get_reranker()
        pairs = [(question[:200], doc.page_content[:400]) for doc, _ in candidates]
        rerank_scores = reranker.predict(pairs, show_progress_bar=False)
    except Exception:
        rerank_scores = [doc_score for _, doc_score in candidates]

    scored = []
    for i, (doc, _) in enumerate(candidates):
        rs = float(rerank_scores[i]) if i < len(rerank_scores) else 0.0
        scored.append((doc, rs))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def law_rag_tool(question: str) -> str:
    try:
        results = _hybrid_rerank(question, top_k=5, vec_k=20, bm25_k=20)

        if not results:
            return "法典检索未找到相关条文，建议联网搜索获取最新司法解释。"

        output = ["## 法典检索结果（混合召回+重排精排）\n"]
        for i, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get("source", "")
            output.append(f"**[{i}]** 来源: {source} | 精排得分: {score:.3f}")
            output.append(f"> {doc.page_content[:400]}")
            output.append("")

        return "\n".join(output)
    except Exception as e:
        return f"法典检索失败: {str(e)}"


# ======================== 工具注册 ========================

def build_tools() -> List:
    return [
        StructuredTool.from_function(func=case_classify_tool, name="case_classify", description="案件类型智能分类：根据案件描述识别类型（民事/刑事/行政）、案由、管辖法院、举证责任和诉讼时效", args_schema=CaseClassifyInput),
        StructuredTool.from_function(func=risk_assess_tool, name="risk_assess", description="法律风险评估：识别法律场景中的潜在风险点、风险等级和潜在后果", args_schema=RiskAssessInput),
        StructuredTool.from_function(func=doc_template_tool, name="doc_template", description="法律文书模板生成：生成起诉状、离婚协议、借款合同、劳动合同、租赁合同、授权委托书等标准模板", args_schema=DocTemplateInput),
        StructuredTool.from_function(func=search_tool, name="search", description="百度搜索：查询最新法律法规动态、司法解释", args_schema=SearchInput),
        StructuredTool.from_function(func=law_rag_tool, name="law_rag", description="法典检索：向量+关键词混合召回 + CrossEncoder精排，基于民法典/刑法典等法典检索法律条文", args_schema=LawRAGInput),
    ]
