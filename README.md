# 🏭 SmartMfg RAG - 智能制造知识库助手

这是一个基于 **RAG (检索增强生成)** 技术的企业级问答系统，专为处理工业制造领域的 PDF 文档设计。采用 **FastAPI + Streamlit** 前后端分离架构，并支持 **Docker 一键部署**。

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## ✨ 项目亮点

- **架构解耦**：基于 FastAPI (后端) + Streamlit (前端) 的微服务架构。
- **混合检索**：集成 BGE-M3 (向量检索) + BM25 (关键词检索) + Rerank (重排序)，大幅提升召回准确率。
- **OCR 增强**：集成 PaddleOCR，支持扫描件和图片 PDF 的文字提取。
- **完全容器化**：提供 Docker Compose 配置，无需本地配置 Python 环境。

## 🛠️ 快速开始 (Quick Start)

由于本项目涉及大模型文件和敏感密钥，请严格按照以下步骤进行配置。

### 1. 克隆项目
```bash
git clone [https://github.com/你的GitHub用户名/SmartMfg-RAG.git](https://github.com/你的GitHub用户名/SmartMfg-RAG.git)
cd SmartMfg-RAG
```

### 2. 配置环境 (Env)
本项目需要 API Key 才能运行。请复制示例配置文件：

```bash
# Mac/Linux
cp .env.example .env

# Windows (PowerShell)
copy .env.example .env
```

**编辑 `.env` 文件**，填入你的 DeepSeek 或 OpenAI Key：
```ini
AI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
AI_BASE_URL=[https://api.deepseek.com](https://api.deepseek.com)
```

### 3. 下载模型 (Models)
由于 GitHub 文件大小限制，你需要手动下载嵌入模型和重排序模型，并放入 `model_cache` 目录。

- **Embedding 模型**: [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
- **Reranker 模型**: [BAAI/bge-reranker-m3](https://huggingface.co/BAAI/bge-reranker-m3)

**目录结构需保持如下：**
```text
SmartMfg-RAG/
├── model_cache/
│   ├── bge-m3/          <--在此处解压 embedding 模型
│   └── bge-reranker-m3/ <--在此处解压 reranker 模型
```

### 4. 启动服务 (Docker)
确保你已安装 Docker Desktop，然后在项目根目录执行：

```bash
docker compose up -d --build
```

等待构建完成后，访问前端页面：
👉 **http://localhost:8501**

## 📚 目录结构说明

- `app/`: 后端 FastAPI 核心逻辑
- `frontend/`: 前端 Streamlit 界面逻辑
- `data/`: ChromaDB 向量数据库持久化目录
- `docker-compose.yml`: 容器编排文件

## 🤝 贡献
欢迎提交 Issue 或 Pull Request！