# 目录结构
# RAG_Prac_CS/
# ├── .env                  # 环境变量 (API Key, Local Paths)
# ├── requirements.txt      # 依赖清单 (核心修正点)
# ├── app/                  # 后端 (FastAPI)
# │   ├── config.py         # 配置加载
# │   ├── main.py           # API 入口
# │   ├── core.py           # 核心业务 (RAG Logic)
# │   ├── schemas.py        # 数据契约 (Pydantic v2)
# │   └── utils/            # 工具类
# │       ├── ocr.py        # OCR 引擎
# │       └── bm25.py       # BM25 检索
# └── frontend/             # 前端 (Streamlit)
#     └── ui.py             # 界面逻辑

# app/config.py
# 必须在导入 os 之前加载 dotenv，否则后续的 os.getenv 拿不到值
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

class Settings:
    # --- 基础服务配置 ---
    # 项目根目录 (假设 app/ 是当前文件的父级)
    BASE_DIR = Path(__file__).resolve().parent.parent
    
    # API 服务设置
    API_TITLE: str = "SmartMfg RAG Enterprise API"
    API_VERSION: str = "2.0.0"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))

    # --- 模型路径 (Docker 容器内路径) ---
    # 默认值适配 Docker 环境，本地调试时可通过 .env 覆盖
    MODEL_PATH: str = os.getenv("MODEL_PATH", "/app/model_cache/bge-m3")
    RERANKER_PATH: str = os.getenv("RERANKER_PATH", "/app/model_cache/bge-reranker-base")
    
    # --- 数据库路径 ---
    DB_PATH: str = os.getenv("DB_PATH", "/app/data/chroma_db")
    DB_NAME: str = "smartmfg_knowledge"
    BM25_PATH: str = os.path.join(DB_PATH, "bm25.pkl")

    # --- LLM 服务 ---
    # 这里不给默认值，强制要求环境变量提供，否则运行时报错(或者由逻辑处理)
    AI_API_KEY: str = os.getenv("AI_API_KEY", "")
    AI_BASE_URL: str = os.getenv("AI_BASE_URL", "https://api.deepseek.com")
    LLM_MODEL_NAME: str = "deepseek-chat"

    # --- RAG 参数 ---
    DEFAULT_TOP_K: int = 20    # 粗排召回数量
    RERANK_TOP_K: int = 3      # 精排最终数量
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    
    # --- OCR 开关 ---
    # ENABLE_OCR: bool = os.getenv("ENABLE_OCR", "True").lower() == "true"
    ENABLE_OCR: bool = str(os.getenv("ENABLE_OCR", "True")).lower() == "true"

# 单例模式
settings = Settings()

# 简单的检查，防止由配置引起的启动失败
if not settings.AI_API_KEY:
    print("⚠️ 警告: 未检测到 AI_API_KEY，LLM 功能将无法使用。")