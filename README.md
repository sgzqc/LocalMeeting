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

## X-ASR 后端

识别后端基于 X-ASR 960 ms Zipformer2、FireRedVAD 和 sherpa-onnx，保持与 X-ASR `x-asr-live-demo/live_asr.py` 相同的核心处理链：

- 16 kHz 单声道 float32 音频和 512-sample VAD 窗口；
- FireRedVAD 下降沿断句；
- 0.7 秒 preroll 句首回补和 1.0 秒 tail padding；
- X-ASR Zipformer2 greedy-search 流式解码；
- 中文 BPE 空格规范化。

服务启动时会预加载 X-ASR 和一个 FireRedVAD 会话。浏览器只在收到后端 `ready` 且完成采样率协商后才开始采音，避免初始化期间堆积音频。FireRedVAD 在会话结束时 reset 并返回预热池。

模型目录需包含 `asr/{encoder,decoder,joiner}-960ms.onnx`、`asr/tokens.txt`、`firered_vad/model.pth.tar` 和 `firered_vad/cmvn.ark`。

开发环境会先在 `models/x-asr/` 查找，未找到时使用已验证的 `C:\Code\X-ASR\X-ASR-zh-en\deployment\x-asr-live-demo\models`。其他环境应通过环境变量明确指定。

用于本地回归验证的示例音频保存在 `tests/fixtures/audio/`，不参与应用运行。

## 配置

复制 `.env.example` 为 `.env`，配置：

- `OPENAI_BASE_URL`
- `OPENAI_API_KEY`
- `OPENAI_MODEL_NAME`
- `X_ASR_MODEL_DIR`
- `X_ASR_VAD_DIR`

`.env`、上传的音频和 Python 缓存均已加入 `.gitignore`。

## 音频说明

实时会议使用 AudioWorklet 在浏览器音频线程采集 PCM，再发送给后端。后端使用 SoXR HQ 流式重采样器转为 16 kHz，其滤波状态会跨浏览器数据块保留，不依赖 FFmpeg。上传文件模式通过 FFmpeg 解码成 16 kHz 单声道 PCM，再进入同一套 X-ASR 流式状态机。
