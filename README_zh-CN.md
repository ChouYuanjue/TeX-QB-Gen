<div align="center">

[English](README.md)| 简体中文

<img src="assets/banner.jpg" width="640px"  alt="TEXQBGEN"/>  

</div>

# TeX Question Bank Generator (TeX-QB-Gen)

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/ChouYuanjue/TeX-QB-Gen)
![GitHub commit activity](https://img.shields.io/github/commit-activity/t/ChouYuanjue/TeX-QB-Gen)
![GitHub License](https://img.shields.io/github/license/ChouYuanjue/TeX-QB-Gen)

> 从教科书直接生成题库。

利用 OpenRouter 提供的多模态/语言模型与本地 OCR 能力，从图片、网页或 PDF 中抽取数学题目与解答，并生成统一格式的 TeX 题库。每道题会生成独立的 TeX 文件，并由 `master.tex` 汇总方便批量编译。

## 背景

这是一个为了赚取 NJU 劳育时长而诞生的产物。当教授要求用 TeX 整理题库来换取劳育时长时，我们为什么不采用更快的方案呢？

> **注**：目前`tests/`目录下所有测试文件均为能够成功处理的示例。

## 功能亮点

- **多种输入渠道**：图片、PDF（自动判断文字版/扫描版）、URL；对 Math Stack Exchange / MathOverflow 采用 API 批量抓取。
- **智能题目拆分**：文字 PDF 通过关键字定位题目边界；扫描版只对疑似题目页调用 OCR 或多模态模型。
- **答案策略**：保留原始答案/解答；遇到“略”“显然”等极简回答时自动调用 LLM 生成 `Solution (by LLM)`，并注明“可能不准确”。
- **统一输出**：每题生成包含 `Exercise`、`Answer`、`Solution`、`Solution (by LLM)` 的 `.tex` 文件，附加来源元数据注释；自动生成 `master.tex`。
- **缓存与重试**：OpenRouter 请求写入磁盘缓存，自动处理限流、重试，减少重复调用。
- **可扩展配置**：通过 `.env` 覆盖模型、缓存、StackExchange API Key、Tesseract 路径等。

## 快速上手

### 1. 准备环境

```powershell
python -m venv .venv
\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. 配置密钥

复制 `.env.example` 为 `.env` 并填好以下内容：

- `OPENROUTER_API_KEY`：在 [OpenRouter](https://openrouter.ai/) 申请的 API Key。
- `STACKEXCHANGE_KEY`（可选）：提升 StackExchange API 配额。
- `TESSERACT_CMD`（可选）：Windows 上 Tesseract 可执行文件路径。
- `TEXBANK_DEFAULT_LANGUAGE`（可选）：默认输出语言，支持 `auto`（保持原文）、`zh`（中文）、`en`（英文）。
- `TEXBANK_OCR_ENGINE`（可选）：OCR 引擎，支持 `tesseract`（默认）或 `paddle`（适用于中文）。

### 3. Tesseract OCR（本地备用）

- Windows：可通过 [官方安装包](https://github.com/UB-Mannheim/tesseract/wiki) 或 Chocolatey 安装。
- 安装后将安装目录加入 `PATH`，或在 `.env` 中设置 `TESSERACT_CMD`。

### 4. 运行命令

```powershell
$env:PYTHONPATH = "src"
python -m texbank.cli --input examples\sample.pdf --out out
```

命令参数：

| 参数 | 说明 |
| ---- | ---- |
| `--input/-i` | 支持多个输入；使用 `@file` 读取列表 |
| `--out/-o` | 输出目录 |
| `--keyword/-k` | StackExchange 搜索关键字 |
| `--max-items/-m` | StackExchange 题目数量上限 |
| `--site/-s` | `math` 或 `mathoverflow` |
| `--no-llm-solution` | 禁用自动生成解答 |
| `--omit-answer-field` | 多模态抽取时仅请求 `exercise` 和 `solution` 字段，不要求返回 `answer` |
| `--language/-l` | 输出语言（`auto`/`zh`/`en`），默认 `auto` |
| `--paired-sequence` | 配置题目/答案配对的 PDF 标签模板及范围，例如 `{chapter}.{section}.{n}|chapter=1-5|section=1-3` |
| `--paired-start` | 指定配对标签遍历的起始题号 |
| `--paired-max-gap` | 同一前缀下连续缺失多少次题号后停止遍历 |
| `--paired-max-questions` | 限制单次配对遍历最多提取的题目数量 |
| `--paired-max-pages` | 单个题号最多抓取的不同页面数量（自动追加后一页不计入此限制） |
| `--paired-prefix-limit` | 当前缀占位符未显式给定范围时的默认遍历上限 |
| `--paired-latest-only` | 当题号在多页命中时，仅使用最后一次出现的页面及其后一页进行抽取 |
| `--verbose/-v` | 打印详细进度日志 |
| `--debug` | 打印调试级别日志（包含堆栈） |

执行后，`out/` 目录将包含各题的 `.tex` 文件与 `master.tex`。

当指定语言与原题语言不一致时，pipeline会自动调用 LLM 翻译题干、解答与附加解答，元数据中会记录是否发生翻译。

## 处理流程详解

- **智能页段定位**：文字型 PDF 会先根据扩展关键词（Exercise/Problem/Question/题目/例题等）抽取候选区块，再针对每个区块生成跨页组合（最多 3 组）以确保题干和答案同屏。
- **多模态优先策略**：默认把候选页渲染成图片并批量提交给多模态模型，提示中携带题目摘要引导模型只返回匹配题目；若请求失败，会记录日志并自动回退到本地 OCR 或原始文本解析。
- **扫描版兜底**：对于疑似扫描件的 PDF，按扫描概率挑选页面转图后复用图片流程，避免对整本文档逐页 OCR。
- **题目合并与后处理**：跨页抽取的结果会进行去重拼接，缺失解答时触发 LLM 生成 `Solution (by LLM)`，同时通过语言检测与翻译确保最终输出遵循 `--language` 或 `TEXBANK_DEFAULT_LANGUAGE` 设置。
- **可观测性**：配合 `--verbose`/`--debug` 可实时查看每个区块的页组尝试、回退原因、翻译状态等细节，方便定位长耗时或响应失败的瓶颈。


## 关键模块

- `texbank.config`：集中管理模型、缓存、OpenRouter、StackExchange 等设置。
- `texbank.llm_client`：OpenRouter 封装，支持文本/多模态请求、结构化 JSON 解析、重试和磁盘缓存。
- `texbank.pdf_utils`：PDF 文字/扫描判定、关键词切分、扫描页提取图片。
- `texbank.pipeline`：统一管线，处理图片/PDF/URL 输入并输出 `ProblemItem` 列表。
- `texbank.texgen`：将 `ProblemItem` 渲染为 TeX 文件并生成 `master.tex`。
- `texbank.stackexchange`：调用 StackExchange API 搜索与获取完整题目、答案。

## 模型选择建议

- **结构化抽取**：`google/gemini-2.5-flash-lite`（经济且对题目结构敏感）。
- **多模态理解**：`google/gemini-2.5-flash-image-preview`（图片中含公式时表现稳定）。
- **解答生成**：`deepseek/deepseek-chat`（性价比高），复杂推理可切换到 `meta-llama/llama-3.1-70b-instruct`。
- **补充推理/校验**：`deepseek/deepseek-r1` 作为低成本备选。

如需离线/本地模型，可在 `.env` 中替换模型名称并自行部署 HTTP 代理。

## 运行测试

```powershell
$env:PYTHONPATH = "src"
pytest
```

测试通过依赖于 stubbed OpenRouter 客户端，确保基础逻辑正确无须真实网络调用。

## 常见问题

- **429 Too Many Requests**：OpenRouter 限流。请稍后重试或申请更高配额。
- **Tesseract 未找到**：安装后保证 `tesseract.exe` 在 `PATH` 或设置 `TESSERACT_CMD`。
- **StackExchange 返回 502**：API 偶尔波动，重试或缩小请求量。
- **TeX 特殊字符**：渲染前会在注释中转义 `%`，公式部分保持原样。

## TO-DO LIST

- [x] 每处理一个题目就即时写入对应的 TeX 文件。
- [x] 修复中文 PDF 被处理为乱码导致的无法正确分题的问题。
- [x] 生成正确且美观的`master.tex`文件，包含目录、章节甚至参考文献等等。
- [x] 处理明显的错误返回结果。比如一次返回多题结果、未正确返回json、LLM生成的答案实际为Markdown语法等等。
- [x] 对扫描版 PDF 进行测试。
- [x] 对答案出现在书后的情况进行测试。
- [ ] 对包含交换图的情况构建合适的提示词并进行测试。
- [ ] 对MathStackExchange的批量获取以及从URL获取的方案进行完善和测试。
- [ ] 提高处理方案的兼容性，适配更多可能的教科书形式。
- [x] 增加异步，提高处理速度。
- [x] 加入适合中文 OCR 的本地方案。
- [ ] 构建更合适的预处理方案，减少大模型产生的幻觉。
- [ ] 更精细的 PDF 题目切分（基于版式分析或深度学习）。
- [ ] 接入更高精度的多模态模型测试并自动成本对比。
- [ ] 增加 Web UI,可视化生成和管理题库。
- [ ] 为 `Solution (by LLM)` 添加自动校验或符号计算检查。

欢迎提交 issue 与 PR 共同完善。