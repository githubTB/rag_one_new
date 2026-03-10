PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
python3 -c "
from paddleocr import PaddleOCR
print('开始下载模型...')
ocr = PaddleOCR(use_angle_cls=True, lang='ch')
print('✅ 模型下载完成')
"