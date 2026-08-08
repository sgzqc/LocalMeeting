# 会议助手

提供两种工作模式：

- 实时会议：浏览器麦克风流式转录，每 10 秒按新增完整句子更新会议总结，停止时生成最终总结。
- 音频文件：上传录音、后台快速转录，识别完成后手动生成总结。

## 启动

项目默认使用以下 Python 环境：

```powershell
& 'C:\Users\zhaoq\miniconda3\envs\py312\python.exe' -m pip install -r requirements.txt
& 'C:\Users\zhaoq\miniconda3\envs\py312\python.exe' run.py
```

浏览器访问 <http://127.0.0.1:8000>。首次开始会议时，需要允许浏览器使用麦克风。

## 配置

复制 `.env.example` 为 `.env`，配置：

- `OPENAI_BASE_URL`
- `OPENAI_API_KEY`
- `OPENAI_MODEL_NAME`

`.env`、上传的音频和 Python 缓存均已加入 `.gitignore`。

## 音频说明

实时会议直接将浏览器 PCM 音频发送给 sherpa-onnx，不依赖 FFmpeg。上传文件模式通过系统中的 FFmpeg 将 WAV、MP3、M4A、FLAC、AAC、OGG 或 WebM 解码成 16 kHz 单声道 PCM。

