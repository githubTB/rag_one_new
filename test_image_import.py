# 测试图片处理库的导入情况

try:
    from langchain_community.document_loaders import ImageLoader
    print("✅ 成功导入 ImageLoader")
except ImportError as e:
    print(f"❌ 导入 ImageLoader 失败: {e}")

try:
    from PIL import Image
    print("✅ 成功导入 PIL.Image")
except ImportError as e:
    print(f"❌ 导入 PIL.Image 失败: {e}")

try:
    import pytesseract
    print("✅ 成功导入 pytesseract")
    # 测试tesseract命令是否可用
    import subprocess
    result = subprocess.run(['tesseract', '--version'], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ tesseract 命令可用")
        print(f"版本: {result.stdout}")
    else:
        print("❌ tesseract 命令不可用")
        print(f"错误: {result.stderr}")
except ImportError as e:
    print(f"❌ 导入 pytesseract 失败: {e}")
