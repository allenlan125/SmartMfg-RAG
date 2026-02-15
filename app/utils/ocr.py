# app/utils/ocr.py
import os
import numpy as np
from pdf2image import convert_from_path
import pdfplumber
from docx import Document
from app.config import settings

class OCREngine:
    _instance = None
    
    def __init__(self):
        self.ocr_model = None
        if settings.ENABLE_OCR:
            self.initialize_model()

    def initialize_model(self):
        """懒加载 PaddleOCR，避免如果不启用 OCR 还要占内存"""
        try:
            from paddleocr import PaddleOCR
            import logging
            # 关闭 Paddle 的调试日志
            logging.getLogger("ppocr").setLevel(logging.WARNING)
            
            print("👁️ [OCR] 正在加载 PaddleOCR 引擎...")
            # 使用 angular_cls 识别方向
            self.ocr_model = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
            # paddleocr 版本（>=2.7.0）已经移除了 show_log 这个参数，或者将其移动到了其他配置项中，因此直接在初始化时传递它会报错 Unknown argument
            # self.ocr_model = PaddleOCR(use_angle_cls=True, lang="ch")
            print("✅ [OCR] 引擎加载完成")
        except Exception as e:
            print(f"❌ [OCR] 引擎加载失败: {e}")

    def extract_text(self, file_path: str, force_ocr: bool = False) -> list[tuple[int, str]]:
        """
        统一的提取入口
        Returns:List[(page_num, text)]
        """
        ext = os.path.splitext(file_path)[1].lower()
        content = []
        
        try:
            # 1. Word 文档
            if ext == '.docx':
                doc = Document(file_path)
                text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
                if text: content.append((1, text))
                return content

            # 2. PDF 文档
            if ext == '.pdf':
                # A. 尝试软提取 (pdfplumber)
                if not force_ocr:
                    with pdfplumber.open(file_path) as pdf:
                        for i, page in enumerate(pdf.pages):
                            text = page.extract_text()
                            if text and len(text.strip()) > 50:
                                content.append((i + 1, text.strip()))
                
                # B. 硬提取 (OCR)
                # 如果软提取失败(空) 或 被强制开启
                if not content or force_ocr:
                    if not self.ocr_model:
                        self.initialize_model()
                        
                    print(f"   📷 [OCR] 启动视觉识别: {os.path.basename(file_path)}")
                    images = convert_from_path(file_path)
                    for i, img in enumerate(images):
                        img_np = np.array(img)
                        # 兼容性处理：调用 ocr 方法
                        try:
                            # 🔴 修正点：显式移除 cls=True (新版 PaddleOCR 已整合)
                            result = self.ocr_model.ocr(img_np)
                        except Exception as e:
                            print(f"⚠️ OCR Warning page {i}: {e}")
                            continue

                        page_text = ""
                        # if result and result[0]:
                        #     txts = [line[1][0] for line in result[0]]
                        #     page_text = "\n".join(txts)
                        # 🔴 修正点：增加对 result 为 None 的空值判断
                        if result and isinstance(result, list) and len(result) > 0 and result[0]:
                            # Paddle 返回结构: [[[[x,y],..], ("text", conf)], ...]
                            txts = [line[1][0] for line in result[0] if line and len(line) > 1]
                            page_text = "\n".join(txts)
                        
                        if page_text.strip():
                            content.append((i + 1, page_text))
                            
        except Exception as e:
            print(f"❌ 解析错误: {e}")
            
        return content

# 单例导出
ocr_engine = OCREngine()
