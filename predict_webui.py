from fastapi import FastAPI, Request, HTTPException
import numpy as np
from predict_by_pb import vocab, model
import logging
from typing import Optional
import uuid

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
app = FastAPI()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SIZE_PER_SECOND = 32000  # 1s 数据长度
MIN_PREDICT_TIME = 3
MIN_PARTIAL_PROB = 90  

@app.post("/predict")
async def predict(request: Request, trace_id: Optional[str] = None):
    """接收流式上传的 16000Hz 16bit 单声道 PCM 数据。返回语言、置信度、音频时长。

    示例：
    {"lang":"english","prob":97.41055965423584,"duration":3.0}
    
    客户端应以原始二进制流（Content-Type: application/octet-stream）分块上传 PCM 数据。
    """
    if not trace_id:
        trace_id = str(uuid.uuid4())
    logger.info("predict: trace_id=%s", trace_id)
    chunks = []
    total = 0
    min_time = MIN_PREDICT_TIME
    result = None
    async for chunk in request.stream():
        logger.debug("predict: chunk_size=%d", len(chunk))
        if chunk:
            chunks.append(chunk)
        total += len(chunk)
        if total / SIZE_PER_SECOND >= min_time:
            min_time = total // SIZE_PER_SECOND + 1
            res = predict_pcm(chunks)
            if res.get("prob", 0) > MIN_PARTIAL_PROB:
                result = res
                break
            
    if not result:
        result = predict_pcm(chunks)
        
    logger.info("predict: duration=%.1fs lang=%s prob=%.2f trace_id=%s", result["duration"], result["lang"], result["prob"], trace_id)
    return result

def predict_pcm(chunks: list) -> dict:
    data = b"".join(chunks)
    if len(data) == 0:
        raise "no data received"
    
    if len(data) % 2 != 0:
        data += b'\x00'
    
    logger.debug("predict: pcm_size=%d", len(data))
    pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    
    output, prob = model.predict_pb(pcm)
    language = vocab.token_list[output.numpy()]
    prob_percent = float(prob.numpy() * 100)

    return {"lang": language, "prob": prob_percent, "duration": len(data)/SIZE_PER_SECOND}