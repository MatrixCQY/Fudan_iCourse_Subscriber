"""LLM-based course lecture summarization via ModelScope API."""

import time

from openai import OpenAI

from src.runtime import config

SYSTEM_PROMPT = r"""你是一名课程助教，根据课程录音转写和 PPT 文字识别，为学生写自学与期末复习用的听课笔记。
学生最想知道的是"老师本人说了什么"，所以笔记要尽量保留老师的原话，并严格按下面的优先级和结构组织。

一、输出结构：固定三节，用下面三个三级标题，顺序不变，标题文字一字不差。

### 【课程事项与考试信息】
- 这一节每次都要输出。逐条列出老师提到的：考试范围与重点（如"这个要考""期末会出这种题"）、题型与分值、考试形式与要求、作业内容与截止时间、签到/组队/展示等课程要求，以及老师明确说"不考""不用掌握"的内容。
- 每条格式：先写"**类型**：一句话概括"，下一行用引用块给出老师原话。例如：
  - **考试范围**：（一句话概括）
    > ……（老师原话）……
- 输入带时间段标记时，在原话后注明所在时间段，如"（约 30–40 分钟）"；没有时间信息就不写。
- 只写老师确实说过的。本讲完全没有提到这类事项时，只写一行"本讲未提及考试或作业相关事项。"，不要把普通知识点硬写成考点。

### 【老师的讲评与提示】
- 收录老师用自己的话做的讲解和提醒：强调重点、提醒易错点、直观解释和打比方、解题思路与经验、对教材或常见说法的评价、学习建议。
- 以原话为主：每条先用一句话交代语境（老师在讲什么），再用引用块给出原话。一条原话一般 1–4 句，保留老师的语气和措辞。
- 按话题归组，不必按时间顺序。宁可多收录，不要只挑一两句。

### 【课堂内容】
- 按老师讲课的逻辑整理知识点，保持详细的课堂笔记水准：包含定义、推导细节、例题、案例和文献，不要过度概括，也不要逐句复述。
- 用完整段落组织讲解，适量使用列表；可以用 ####、##### 作小标题。
- 老师讲得特别清楚或有独特表述的地方，用引用块保留原话，再接你的整理。
- PPT 文字是课件原文，用来补全公式、定义和修正术语，但主线以老师的讲解为准。

二、"原话"的标准（轻度清洗）
- 引用块（以 > 开头的行）里只能放老师说过的话。可以：修正明显的同音字和识别错误、删掉"嗯""啊""那个"等口头禅和无意义的重复、补上标点。不可以：改写句式、概括、把不同位置的话拼成一句、加入老师没说过的内容。
- 拿不准是不是老师原话的内容，不要放进引用块，改用普通叙述。
- 学生的提问或回答如果有价值，可以在引用前标注"（学生）"。

三、格式要求
- 不要任何客套话，不要输出课程名大标题，直接从"### 【课程事项与考试信息】"开始。
- 只使用三级及以下标题（###、####、#####），禁止使用 # 和 ##。
- 列表的缩进和嵌套都不超过两级。
- 数学公式使用规范的 LaTeX：行内 $...$，行间 $$...$$；公式中不要出现中文。
- 不要保留输入里的时间段分隔线和"【音频转录】""【PPT 文字识别】"等标记；时间信息只按上面的规则写在原话后面。

四、输入材料说明
- 输入是以下两种格式之一：
  A) 带时间轴的分段格式：用 `=== 时间段 mm:ss – mm:ss ===` 分隔每 10 分钟，段内有【音频转录】和【PPT 文字识别】两类内容。
  B) 平铺格式：先是【音频转录（无时间轴）】，后是【PPT 文字识别（按出现顺序）】，每张 PPT 带 `[页 N @ mm:ss]` 标签。
- 音频转录来自语音识别，常有同音字和专有名词错误，要结合学术语境和 PPT 修正；PPT 文字也可能有 OCR 错误。
- PPT 来自屏幕截图，可能混入网页、导航栏、桌面、系统界面等无关文字，忽略这些噪音。
- 禁止捏造录音和 PPT 中都没有出现的内容。"""

class Summarizer:
    """Course lecture summarizer with multi-provider fallback.

    Iterates config.MODEL_PROVIDERS in declared order. Within each provider,
    tries each model in declared order. Returns the first successful result.
    Setting only DASHSCOPE_API_KEY still works because the default
    MODEL_PROVIDERS list ships a modelscope entry that reads it.
    """

    def __init__(self):
        self.providers = config.resolve_model_providers()
        if not self.providers:
            raise ValueError(
                "No model provider available. "
                "Set at least one provider's API key (e.g. DASHSCOPE_API_KEY)."
            )
        self._clients = {
            p["name"]: OpenAI(api_key=p["api_key"], base_url=p["base_url"])
            for p in self.providers
        }

    def _call_llm(self, client: OpenAI, model: str,
                  title: str, content: str,
                  max_tokens: int | None = None, timeout: int = 180) -> str:
        t0 = time.time()
        extra = {"max_tokens": max_tokens} if max_tokens else {}
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    # Length hint covers only 【课堂内容】 (kept at the old ~1:7
                    # signal); the exam-info and teacher-remarks sections are
                    # extra and sized by what the teacher actually said.
                    "content": (
                        f"以下是课程《{title}》的录音转写与 PPT 文字。"
                        f"【课堂内容】一节约 {len(content) // 7} 字；"
                        f"前两节按老师实际所说的内容另计，不受此字数限制。请开始：\n\n{content}"
                    ),
                },
            ],
            # temperature=0.3,
            timeout=timeout,
            **extra,
        )
        if not response.choices:
            raise ValueError("API returned empty choices — likely content filter or quota exceeded")
        result = response.choices[0].message.content
        elapsed = time.time() - t0
        # Token usage helps explain run cost — every provider's billing is
        # token-based, and rate-limit decisions key off prompt size much
        # more than character count.  Some providers (OpenAI-compatible)
        # leave usage None on streaming or error paths, so fall back to a
        # plain "no usage" line so the summary still prints.
        usage = getattr(response, "usage", None)
        if usage is not None:
            print(
                f"[Summarizer] Done ({model}): "
                f"{len(content)} chars input → {len(result)} chars output"
                f" in {elapsed:.0f}s "
                f"(tokens: prompt={getattr(usage,'prompt_tokens','?')}, "
                f"completion={getattr(usage,'completion_tokens','?')})"
            )
        else:
            print(
                f"[Summarizer] Done ({model}): {len(content)} chars input"
                f" → {len(result)} chars output in {elapsed:.0f}s"
            )
        return result

    def summarize(self, title: str, content: str) -> tuple[str, str]:
        """Summarize lecture, trying providers in MODEL_PROVIDERS order.

        Returns (summary, model_used) where model_used is "{provider}/{model}".

        Raises:
            RuntimeError: if all providers/models fail.
        """
        if not content or not content.strip():
            return ("（内容为空）", "")

        errors = []
        for provider in self.providers:
            client = self._clients[provider["name"]]
            for model in provider["models"]:
                model_id = f"{provider['name']}/{model}"
                try:
                    result = self._call_llm(
                        client, model, title, content,
                        max_tokens=provider.get("max_tokens"),
                        timeout=provider.get("timeout") or 180,
                    )
                    return (result, model_id)
                except Exception as e:
                    print(f"[Summarizer] {model_id} failed: "
                          f"{type(e).__name__}: {e}")
                    errors.append(f"{model_id}: {e}")

        raise RuntimeError(
            "All LLM models failed:\n" + "\n".join(errors)
        )
