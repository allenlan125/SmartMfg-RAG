# frontend/ui.py
import streamlit as st
import requests
import json
import os

# 后端 API 地址 (本地调试用)
# 如果是 Docker 部署，这里会从环境变量读取，默认为 localhost
API_BASE_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="RAG 企业版 (CS架构)",
    page_icon="🏭",
    layout="wide"
)

st.title("🏭 智能制造知识库 (生产级重构版)")

# --- 侧边栏：文件上传 ---
with st.sidebar:
    st.header("📄 知识库管理")
    uploaded_file = st.file_uploader("上传新文档 (PDF/Word)", type=["pdf", "docx"])
    use_ocr = st.checkbox("启用 OCR 增强模式", value=True, help="对扫描件或图片PDF启用视觉识别")
    
    if uploaded_file and st.button("开始上传与处理"):
        with st.spinner("文件上传与索引中..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                data = {"use_ocr": str(use_ocr)} # Multipart form data
                
                # 调用后端 /upload 接口
                resp = requests.post(f"{API_BASE_URL}/upload", files=files, data=data)
                
                if resp.status_code == 200:
                    res_json = resp.json()
                    st.success(f"✅ 处理成功！共生成 {res_json.get('chunks_count')} 个切片")
                else:
                    st.error(f"❌ 上传失败: {resp.text}")
            except Exception as e:
                st.error(f"🔌 连接错误: {e}")

    st.divider()
    
    # 健康检查
    if st.button("检查后端连接"):
        try:
            resp = requests.get(f"{API_BASE_URL}/health")
            if resp.status_code == 200:
                st.success(f"后端在线: {resp.json()}")
            else:
                st.error("后端状态异常")
        except:
            st.error("无法连接到后端，请检查 main.py 是否在运行")

# --- 主界面：聊天窗口 ---

# 初始化历史记录
if "messages" not in st.session_state:
    st.session_state.messages = []

# 渲染历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # 如果有引用源，也可以尝试在这里渲染（需要存到 session state）

# 处理用户输入
if prompt := st.chat_input("请输入您的问题，例如：注塑机温度异常怎么处理？"):
    # 1. 显示用户问题
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 调用后端获取回答
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("🧠 大脑思考中 (检索-排序-生成)..."):
            try:
                # 构造请求体 (符合 schemas.ChatRequest)
                payload = {
                    "question": prompt,
                    "history": [
                        {"role": m["role"], "content": m["content"]} 
                        for m in st.session_state.messages[:-1]
                    ],
                    "top_k": 3
                }
                
                # 发送 POST 请求
                response = requests.post(f"{API_BASE_URL}/chat", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data["sources"]
                    
                    # 展示回答
                    st.markdown(answer)
                    
                    # 展示引用源 (折叠显示)
                    if sources:
                        with st.expander(f"📚 参考了 {len(sources)} 个文档片段"):
                            for idx, src in enumerate(sources):
                                st.markdown(f"**[{idx+1}] {src['source']} (Page {src['page']})** `Score: {src['score']:.4f}`")
                                st.caption(src['content'])
                    
                    # 存入历史
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                else:
                    st.error(f"❌ 后端报错: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("🔌 无法连接后端服务，请确认 python app/main.py 正在运行！")
            except Exception as e:
                st.error(f"⚠️ 发生未知错误: {e}")