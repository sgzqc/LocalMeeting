from __future__ import annotations

from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from app.config import Settings


SUMMARY_TEMPLATE = """请根据会议转录生成准确、简洁、结构化的中文会议纪要。
不要虚构转录中未出现的人名、日期、负责人或结论；不确定的信息标记为“待确认”。
严格使用以下结构：
## 会议摘要
用一段话概括会议主题和主要内容。
## 关键讨论与结论
合并重复观点，只保留重要讨论、决定和结论。
## 待办事项
使用简洁列表；尽可能包含事项、负责人和期限，缺失项标记为“待确认”。
## 风险与待确认事项
只列出确实存在的风险、分歧和未决问题；没有则写“无”。
"""


class SummaryService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url or None,
        ) if settings.llm_ready else None

    async def summarize_full(self, transcript: str) -> str:
        if not transcript.strip():
            raise ValueError("暂无可总结的转录内容")
        if not self.client:
            raise RuntimeError("OpenAI 配置不完整")
        response = await self.client.chat.completions.create(
            model=self.settings.openai_model_name,
            messages=[
                {"role": "system", "content": SUMMARY_TEMPLATE},
                {"role": "user", "content": f"会议完整转录：\n{transcript}"},
            ],
            temperature=0.2,
            max_tokens=1200,
            extra_body={"thinking": {"type": "disabled"}},
        )
        return response.choices[0].message.content or ""

    async def stream_full(self, transcript: str) -> AsyncIterator[str]:
        if not transcript.strip():
            raise ValueError("暂无可总结的转录内容")
        if not self.client:
            raise RuntimeError("OpenAI 配置不完整")
        stream = await self.client.chat.completions.create(
            model=self.settings.openai_model_name,
            messages=[
                {"role": "system", "content": SUMMARY_TEMPLATE},
                {"role": "user", "content": f"会议完整转录：\n{transcript}"},
            ],
            temperature=0.2,
            max_tokens=1200,
            stream=True,
            extra_body={"thinking": {"type": "disabled"}},
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta

    async def summarize_incremental(self, previous: str, new_transcript: str) -> str:
        if not self.client:
            raise RuntimeError("OpenAI 配置不完整")
        response = await self.client.chat.completions.create(
            model=self.settings.openai_model_name,
            messages=[
                {"role": "system", "content": SUMMARY_TEMPLATE},
                {
                    "role": "user",
                    "content": (
                        "请结合已有纪要和新增转录，输出更新后的完整纪要。\n\n"
                        f"已有纪要：\n{previous or '暂无'}\n\n新增转录：\n{new_transcript}"
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=1200,
            extra_body={"thinking": {"type": "disabled"}},
        )
        return response.choices[0].message.content or ""
