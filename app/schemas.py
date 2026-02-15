# app/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# --- 基础响应模型 ---
class HealthResponse(BaseModel):
    status: str = Field(..., description="服务状态", example="ok")
    version: str = Field(..., description="API版本")
    components: Dict[str, str] = Field(default={}, description="各组件加载状态")

# --- 聊天相关模型 ---

class ChatMessage(BaseModel):
    role: str = Field(..., description="角色: user 或 assistant", example="user")
    content: str = Field(..., description="消息内容", example="SDN是什么？")

class ChatRequest(BaseModel):
    # question: str = Field(..., description="用户当前的问题", example="注塑机报警怎么处理？")
    # history: List[ChatMessage] = Field(default=[], description="历史对话上下文 (用于指代消解)")
    # top_k: int = Field(default=3, description="最终返回的参考文档数量")
    # use_search: bool = Field(default=True, description="是否启用知识库检索")
    # # 高级参数
    # score_threshold: float = Field(default=-10.0, description="Reranker 分数截断阈值")
    # Pydantic v2 推荐写法
    question: str = Field(..., description="用户问题", examples=["SDN是什么？"]) 
    history: List[Dict[str, str]] = Field(default_factory=list, description="历史记录") # 使用 default_factory
    top_k: int = Field(default=3, ge=1, le=20) # 增加数值约束：>=1, <=20
    use_search: bool = True

class SourceDocument(BaseModel):
    content: str = Field(..., description="文档切片内容")
    source: str = Field(..., description="来源文件名")
    page: int = Field(default=0, description="页码")
    score: float = Field(..., description="相关性得分 (Reranker Logits)")

class ChatResponse(BaseModel):
    answer: str = Field(..., description="LLM 生成的回答")
    rewritten_query: Optional[str] = Field(None, description="经过指代消解后的查询语句")
    sources: List[SourceDocument] = Field(default=[], description="引用的参考文档")
    process_time: float = Field(default=0.0, description="处理耗时(秒)")

# --- 文件上传相关模型 ---

class UploadResponse(BaseModel):
    filename: str
    status: str = "success"
    message: str
    chunks_count: int = 0