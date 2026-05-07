# knowledge-demo · 个人知识库笔记整理 Agent

基于向量检索与 LLM 的本地知识库工具，将散落的 Markdown 笔记转化为可语义检索的结构化知识库。

## 功能

- **笔记导入** — 递归扫描 Markdown 文件目录，自动切块后存入向量数据库
- **自动标引** — 对笔记批量打标签、分类、提取关键词（LLM / 本地规则双模式）
- **语义检索** — 自然语言提问，向量召回最相关内容
- **知识生成** — 基于知识库的问答对话、按主题/分类生成周报或专题总结
- **流式输出** — LLM 回答逐字呈现，交互更流畅

## 快速开始

### 安装

```bash
git clone https://github.com/moyu-by/knowledge-demo
cd knowledge-demo
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> 国内用户需设置 HuggingFace 镜像：
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com
> ```

### 配置 LLM（可选）

设置环境变量即可，不配置时自动降级为本地规则模式：

```bash
export ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
export ANTHROPIC_API_KEY=sk-xxxx
```

支持的模型：`deepseek-v4-flash`（默认）、`deepseek-v4-pro`，或其他 Anthropic API 兼容的服务。

也可以通过 `--model` 参数临时切换：
```bash
python main.py query "问题" --model deepseek-v4-pro
```

### 导入笔记

```bash
python main.py import ./data/notes
# 输出: 导入完成：N 篇笔记，N 个片段
```

### 自动标引

```bash
python main.py process
# 输出: 处理完成（模式: LLM/本地规则）: 成功 N, 失败 0
```

### 使用

```bash
# 自然语言问答（流式输出）
python main.py query "Java 集合框架有哪些实现"

# 周报
python main.py report --days 7

# 专题知识总结
python main.py topic "分布式系统"

# 浏览知识库
python main.py list          # 列出所有笔记
python main.py tags          # 查看标签和分类统计
python main.py stats         # 知识库统计

# 管理
python main.py delete        # 交互式删除笔记索引
python main.py delete --all  # 清空知识库索引（不删源文件）
```

## 配置

编辑 `config.yaml`：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `kb.notes_dir` | 笔记目录路径 | `./data/notes` |
| `embedding.model` | 嵌入模型 | `BAAI/bge-small-zh-v1.5` |
| `llm.model` | LLM 模型名 | `deepseek-v4-flash` |
| `llm.api_base` | API 地址 | `ANTHROPIC_BASE_URL` 环境变量 |
| `llm.api_key` | API Key | `ANTHROPIC_API_KEY` 环境变量 |
| `chunking.max_tokens` | 笔记切块大小 | 1000 |
| `retrieval.top_k` | 检索返回片段数 | 10 |

## 架构

```
┌─────────────────────────────────────────────────┐
│                   CLI (main.py)                  │
├─────────┬──────────┬──────────┬─────────────────┤
│  import  │ process  │  query   │  report / topic │
├─────────┴──────────┴──────────┴─────────────────┤
│   loader.py → vector_store.py → retriever.py    │
│              ↕ processor.py / generator.py       │
├─────────────────────────────────────────────────┤
│   ChromaDB (向量存储)  │  sentence-transformers   │
│   BGE-small-zh (嵌入)  │  Anthropic SDK (LLM)     │
└─────────────────────────────────────────────────┘
```

## 数据流

1. **导入**：扫描目录 → 解析 Markdown / frontmatter → 切块 → 向量化 → 存入 ChromaDB
2. **标引**：读取未处理笔记 → 批量调用 LLM（或本地规则）→ 提取标签/分类/关键词 → 更新元数据
3. **检索**：用户提问 → 向量检索 top-k 片段 → 拼接上下文 → LLM 生成回答（流式）
4. **生成**：按分类聚合笔记 → LLM 总结 → 输出周报 / 专题知识

## 项目结构

```
knowledge-demo/
├── main.py              # CLI 入口
├── config.yaml          # 配置文件
├── requirements.txt     # Python 依赖
├── run.sh               # 启动脚本（含国内镜像设置）
├── src/
│   ├── models.py        # 数据模型
│   ├── config.py        # 配置加载
│   ├── loader.py        # Markdown 读取与切块
│   ├── vector_store.py  # ChromaDB 向量存储封装
│   ├── processor.py     # 标签/分类/关键词提取
│   ├── retriever.py     # 语义检索
│   └── generator.py     # 问答/周报/专题生成
└── kb_storage/          # 向量数据库持久化（自动生成）
```

## 依赖

- Python ≥ 3.10
- chromadb — 本地向量数据库
- sentence-transformers — 文本嵌入
- anthropic — Anthropic API / DeepSeek 兼容接口
