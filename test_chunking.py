#!/usr/bin/env python3
"""
测试文档分块逻辑
"""

from langchain_core.documents import Document
from pipeline import split_by_heading

# 创建一个长文本测试文档
def create_test_doc():
    # 创建一个长度为80000的文本
    long_text = "测试文本 " * 40000  # 80000字符
    print(f"创建测试文档，长度: {len(long_text)}字符")
    
    return [Document(
        page_content=long_text,
        metadata={"source": "test.txt", "type": "Text"}
    )]

# 测试分块逻辑
def test_chunking():
    print("=== 测试文档分块逻辑 ===")
    
    try:
        # 创建测试文档
        docs = create_test_doc()
        
        # 分块
        splits = split_by_heading(docs)
        
        # 检查所有块的大小
        print(f"分块结果: {len(splits)}块")
        
        # 检查是否有超过60000字符的块
        oversized_chunks = [len(chunk.page_content) for chunk in splits if len(chunk.page_content) > 60000]
        if oversized_chunks:
            print(f"❌ 发现超大块: {oversized_chunks}")
            return False
        else:
            print(f"✅ 所有块大小正常，最大块: {max(len(chunk.page_content) for chunk in splits)}字符")
            
            # 模拟app.py中的额外检查和分割
            for i, split in enumerate(splits):
                if len(split.page_content) > 60000:
                    # 极端情况，直接按字符数分割
                    content = split.page_content
                    meta = split.metadata
                    
                    # 替换当前块为多个小块
                    new_chunks = []
                    start = 0
                    while start < len(content):
                        end = min(start + 60000, len(content))
                        # 尝试在句子边界结束
                        if end < len(content):
                            # 查找最近的句子结束符
                            end_puncts = [content.rfind(p, start, end) for p in ['.', '!', '?', '。', '！', '？']]
                            end_punct = max([p for p in end_puncts if p > start])
                            if end_punct > start + 60000 * 0.8:  # 确保至少有80%的内容
                                end = end_punct + 1
                    
                        new_chunks.append(Document(page_content=content[start:end].strip(), metadata=meta))
                        start = end
                    
                    # 替换当前块
                    splits[i:i+1] = new_chunks
            
            # 再次检查
            oversized_chunks = [len(chunk.page_content) for chunk in splits if len(chunk.page_content) > 60000]
            if oversized_chunks:
                print(f"❌ 额外检查后仍发现超大块: {oversized_chunks}")
                return False
            else:
                print(f"✅ 额外检查后所有块大小正常，最大块: {max(len(chunk.page_content) for chunk in splits)}字符")
                print(f"   最终分块数: {len(splits)}")
                return True
                
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    success = test_chunking()
    exit(0 if success else 1)
