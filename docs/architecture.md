# 架构说明

README 里的架构图讲的是「有哪些节点」，这份文档讲的是**每个节点的契约**——
它读什么、写什么、为什么这么设计。

调 bug 的时候翻这份，比翻代码快。

---

## 状态 Schema

### `OverAllState`（主图）

| 字段 | 类型 | 谁写 | 为什么这么设计 |
|---|---|---|---|
| `topic` | `str` | CLI | 入口输入 |
| `plan` | `ResearchPlan \| None` | planner / human_review | 结构化，不是字符串 |
| `approved` | `bool` | human_review | 单写者，无 reducer |
| `feedback` | `str` | human_review | 打回理由，planner 会读它 |
| `sub_question` | `str` | Send payload | 传给子图的当前子问题 |
| `round` | `int` | Send payload | 第几轮检索 |
| **`evidence`** | `Annotated[list[Evidence], operator.add]` | researcher 分支 ×N | **并行分支的唯一合并出口** |
| **`iteration`** | `int` | aggregator | **计数器，绝对不能加 reducer** |
| `gap_verdict` | `GapVerdict \| None` | gap_checker | 自我反思的产物 |
| `report` | `str` | writer | 最终产出。**qa_followup 不碰它** |
| `report_path` | `str` | writer | 落盘位置 |
| `followup_question` | `str` | CLI | 入口路由靠它区分「新研究 / 追问」 |
| `messages` | `Annotated[list[AnyMessage], add_messages]` | qa_followup | 追问历史 |

**三条铁律**（写在 [graph/state.py](../graph/state.py) 顶部）：

1. 被多个并行分支写入的 key **必须**有 reducer，否则 `InvalidUpdateError`
2. 用 `operator.add` 时，分支返回**必须包成列表**：`{"evidence": [e]}`
3. 计数器**不能**用 `operator.add`，否则读出来是 `[1]` 而不是 `1`

### `ResearcherState` / `ResearcherOutput`（子图）

子图内部状态里除了 `evidence` 还有 `messages`（ReAct 对话历史）和 `tool_rounds`。
这些**父图看不见**——`output_schema=ResearcherOutput` 只暴露 `evidence`。

**为什么必须声明 output_schema**：不声明的话，子图会把整个最终状态交回父图，
包括只打算当入参的 `topic`。N 个并行分支各写一次，父图又没有它的 reducer，
直接 `InvalidUpdateError`。

---

## 节点契约

### `planner`
- **读**：`topic`、`feedback`
- **写**：`plan`
- **要点**：`with_structured_output(ResearchPlan)`；被打回时把 `feedback` 带进提示词
- **重试**：`llm_retry_policy()`（结构化输出可能验证失败）

### `human_review`
- **读**：`plan`
- **写**：`plan`（可能被用户编辑）、`approved`、`feedback`
- **要点**：**`interrupt()` 之前的代码会重跑**，所以那之前不能有副作用
- **重试**：**不加** —— 不调用模型，而且重试「等人做决定」没有意义

### `researcher`（子图）
- **入**：`topic`、`sub_question`、`round`
- **出**：`evidence`
- **内部**：`prepare → agent ⇄ tools → summarize`，最多 4 轮工具调用
- **要点**：轮次用尽时换**不绑工具**的模型，而不是在路由里改道 ——
  后者会留下「有 tool_calls 却没有结果」的非法消息序列

### `aggregator`
- **读**：`evidence`
- **写**：`evidence`（用 **`Overwrite`** 替换）、`iteration`
- **要点**：
  - 用 `dict` 而不是 `set` 登记来源（set 顺序有哈希随机化，报告不可复现）
  - 返回值必须 `Overwrite(...)`，否则是追加不是替换
  - 轮次计数放这里，因为 aggregator 每轮**恰好执行一次**

### `gap_checker`
- **读**：`topic`、`plan`、`evidence`（**只看 claim，不看 quote**）
- **写**：`gap_verdict`
- **要点**：按用途裁剪输入 —— 判断「够不够」不需要逐字原文

### `writer`
- **读**：`topic`、`evidence`、`gap_verdict`
- **写**：`report`、`report_path`
- **要点**：
  - 链末尾必须 `| StrOutputParser()`，否则 `stream()` 给的是 `AIMessageChunk`
  - 生成后立刻**机器校验**引用编号，不信任模型自觉
  - 有未解决缺口时，在报告里明写「局限与未决问题」

### `qa_followup`
- **读**：`topic`、`report`、`evidence`、`messages`、`followup_question`
- **写**：**只有** `messages`
- **要点**：**绝不碰 `report`** —— 否则用户上次的报告会被追问覆盖，检查点里也一起没了

---

## 控制流（全部在 `graph/routers.py`）

| 路由 | 位置 | 出口 |
|---|---|---|
| `route_entry` | START | `planner`（新研究）/ `qa_followup`（追问）|
| `route_after_review` | human_review | `planner`（打回）/ **`Send` 列表**（通过）|
| `route_after_gap` | gap_checker | `writer`（三种情况）/ **`Send` 列表**（补搜）|

### `route_after_gap` 的四条出口

```
① 证据充分                      → writer
② 到达 iteration 上限           → writer（诚实降级，报告里声明缺口）
③ 判了不足但没给可搜的问句       → writer
④ 判了不足 + 有轮次 + 有问句     → Send 列表（唯一的循环出口）
```

**出口 ③ 是必需的**：返回空列表会让图**静默终止** —— 不报错、没有报告、
用户看到的是「运行完成」。这是实际踩过的坑。

---

## 循环的三重熔断

| 熔断 | 位置 | 作用 |
|---|---|---|
| `max_iterations` | `config/research.yml` | 业务上限：最多补搜几轮 |
| `recursion_limit` | `config/research.yml` | LangGraph 硬顶：任何意外的边循环都会撞上 |
| 空 `follow_up_queries` 检查 | `route_after_gap` | LLM 没给出可搜问句时强制退出 |

---

## 持久化

```
utils/checkpointer.py
    ├─ memory  → InMemorySaver   进程内，退出即忘
    └─ sqlite  → SqliteSaver     data/checkpoints/research.sqlite
```

**两个必须的细节**：

- 用 `SqliteSaver(conn)` 而不是 `SqliteSaver.from_conn_string(...)` ——
  后者是上下文管理器，`with` 退出即关连接
- `check_same_thread=False` —— 并行分支跑在线程池里

**`thread_id` 一个概念吃三件事**：断点续跑、人工审核跨进程恢复、多轮追问。

---

## 可观测性

- **CLI**：`stream_mode=["updates","custom"]` + `subgraphs=True`，
  custom 事件带 namespace 标签区分并行分支
- **LangSmith**：环境变量驱动，无需改代码
- **LangGraph Studio**：`langgraph dev`

---

## 关键文件速查

| 想改什么 | 改哪里 |
|---|---|
| 控制流 | `graph/routers.py` |
| 节点行为 | `graph/nodes.py` |
| 状态字段 | `graph/state.py` |
| 图的装配 | `graph/build.py` |
| LLM 输出格式 | `schemas/research.py` |
| 提示词 | `prompts/*.txt` |
| 参数（轮次/并发/成本） | `config/*.yml` |
| 重试策略 | `utils/retry.py` |
| 持久化 | `utils/checkpointer.py` |
