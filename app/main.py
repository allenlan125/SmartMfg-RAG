# app/main.py
import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from tempfile import NamedTemporaryFile

from app.schemas import ChatRequest, ChatResponse, HealthResponse, UploadResponse, SourceDocument
from app.config import settings
from app.core import rag_service

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok", version=settings.API_VERSION)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    RAG 对话接口
    """
    try:
        result = rag_service.chat(
            query=request.question,
            history=request.history,
            top_k=request.top_k
        )
        
        # 转换为 Pydantic 模型
        sources = []
        for i, (doc, meta, score) in enumerate(zip(result['docs'], result['metas'], result['scores'])):
            sources.append(SourceDocument(
                content=doc,
                source=meta.get('source', 'unknown'),
                page=meta.get('page', 0),
                score=score
            ))
            
        return ChatResponse(
            answer=result['answer'],
            sources=sources,
            # 未来可接入 rewrite 逻辑
            rewritten_query=request.question 
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    use_ocr: bool = Form(False) # 从 Form Data 获取
):
    try:
        # 保存到临时文件
        suffix = os.path.splitext(file.filename)[1]
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
            
        # 调用核心处理逻辑
        try:
            count = rag_service.process_upload(tmp_path, file.filename, use_ocr)
        finally:
            os.unlink(tmp_path) # 清理临时文件
            
        return UploadResponse(
            filename=file.filename,
            status="success",
            message=f"成功处理并入库 {count} 个切片",
            chunks_count=count
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 本地调试启动逻辑
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)