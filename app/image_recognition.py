"""Explicit image transcription using the configured model; output requires review."""
import base64
import binascii
import tempfile
from pathlib import Path
from . import model, settings, uploads

PROMPT = ('请忠实转写图片中可见的数学题目、文字和手写解题步骤，保留原有顺序、题号和错误。'
          '数学公式用 LaTeX 表示；无法辨认的地方写 [无法辨认]，不要猜测、纠正、补全或解题。'
          '只输出转写内容。')

def recognize(name, encoded):
    if not isinstance(name, str) or not isinstance(encoded, str) or len(encoded) > uploads.MAX_BYTES * 2:
        raise ValueError('需要不超过 5 MB 的图片。')
    ext = uploads._extension(name)
    if ext not in ('.png', '.jpg', '.webp'):
        raise ValueError('仅支持 PNG、JPEG 或 WebP 图片。')
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError('图片编码无效。') from exc
    if not raw or len(raw) > uploads.MAX_BYTES:
        raise ValueError('需要不超过 5 MB 的图片。')
    _, mime = uploads._sniff(raw, ext)
    config = settings.resolve()
    if config['backend'] == 'http':
        url = 'data:' + mime + ';base64,' + base64.b64encode(raw).decode('ascii')
        result = model._generate_http(PROMPT, 120, config, image_data_url=url)
    elif config['backend'] == 'codex-cli':
        with tempfile.TemporaryDirectory(prefix='math-ocr-') as folder:
            path = Path(folder) / ('image' + ext)
            path.write_bytes(raw)
            result = model._generate_codex(PROMPT, 120, config, image_path=path)
    else:
        raise model.ModelError('当前 DeepSeek Harness 后端尚未接入图片识别，请切换到支持图片输入的 HTTP 模型或 Codex。')
    return {'text': result['raw'][:uploads.MAX_EXTRACT_CHARS], 'warning': '识别结果可能有误，请逐项核对公式、符号和步骤后再填入。'}
