<div align="center">

# 🧾 财务报销审核平台

**规则引擎优先 + LLM 兜底的智能报销审核系统** · 逐条核验 · 风险拦截 · 全链路可解释可审计

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11+-ffd43b?style=flat-square&logo=python&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white">
  <img alt="VectorStore" src="https://img.shields.io/badge/VectorStore-SQLite%2BFAISS-9b59b6?style=flat-square">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green?style=flat-square">
  <img alt="RAG" src="https://img.shields.io/badge/Multi--Modal-RAG-3b82f6?style=flat-square">
</p>

**票据识别 · 发票查重 · 供应商黑名单 · 金额交叉核对 · LLM 兜底裁决**

</div>

---

## 📖 项目简介

一个面向**企业报销审核**场景的智能系统。员工提交报销单据（**发票凭证 / 报销信息 / 手填条目**三要素），系统调用**规则引擎**逐条核验，命中硬性规则（发票唯一性、供应商黑名单、金额交叉核对、要素完整性、行程一致性等）直接给出判定；规则无法覆盖的模糊情形再由 **LLM 兜底裁决**。

> **为什么做**：传统报销审核依赖人工逐单核对，效率低、标准不一、且难以追溯。本项目把**确定性规则**作为第一道防线——规则能判定的绝不让模型拍板，从根上解决"AI 审核不可解释、不可控"的痛点。

**三条设计底线**
1. **规则优先**：机械性、确定性的判定全部由规则引擎完成，**不经 LLM**（离线基准可复现，见「评估与量化」）；
2. **逐条可解释**：每个条目单独给出「通过 / 异常 / 需人工」结论 + 理由 + **归因标签**（规则 or 模型）；
3. **安全可控**：模型只输出审计**意图**，实际数据访问与执行权归 Runtime，经七道闸门校验后才放行（**未注册的操作默认拒绝**）。

---

## ✨ 核心特性

### 🔍 规则优先、LLM 兜底
| 机制 | 说明 |
|------|------|
| **规则引擎驱动** | 发票唯一性（重复报销拦截）、供应商黑名单、金额交叉核对、要素完整性与一致性、行程逻辑校验等全部由确定性规则判定 |
| **逐条判定** | 对每个条目分别给出「通过 / 异常 / 需人工」结论与理由，而非一个笼统结论 |
| **LLM 仅兜底** | 仅当规则无法覆盖（如解释性例外、模糊描述）才调用大模型，结果单独标注来源 |

### 🛡️ 可控与可审计
- **权限闸门**：模型只输出审计**意图**，执行权归 Runtime。七道闸门依次校验——**① 工具白名单（未注册即拒）② 身份 ③ 角色 / Scope ④ 参数 Schema（含路径越界）⑤ 风险分级 ⑥ 人工审批 ⑦ 限流 / 预算**，结论为 `ALLOW / CONFIRM / DENY` 三档
- **执行后治理**：返回结果最小化脱敏（递归覆盖手机号 / 证件号 / 银行卡 / 金额）、操作全程审计留痕、**幂等与预算落 SQLite（跨进程重启仍生效）**、写操作可登记回滚钩子
- **防 Prompt 注入**：输入与检索结果**双向**扫描 + 高危指令隔离 + 分界符号，阻断"把模型当后门"的注入攻击

### ⚙️ 工程健壮性
- **客户端断开即止损**：流式端点检测到客户端断开后**立即停止 LLM 生成**，不再为已离开的用户烧 token
- **流式不阻塞事件循环**：同步 LLM 流经「后台线程 + 队列」转为异步迭代，流式期间其他并发请求不受影响
- **内存有界**：限流器带定期 sweep 与用户数上限、上传任务表按上限裁剪、权限闸门内部状态均设硬上限与令牌有效期
- **容器化部署**：多阶段构建 `Dockerfile` + `docker-compose`（镜像只含代码与环境，数据卷挂载持久化）

### 📄 多模态数据底座
- 内置多模态 RAG 底层：多格式文档解析（PDF / Word / Excel / PPT / 图片 / OCR）、**双路检索（向量 + BM25）+ RRF 融合**、精排、冲突检测
- **票据图片识图解析**：提取发票号、金额、供应商、日期等结构化字段
- **自研确定性向量库**：SQLite（WAL，事务提交即落盘）+ FAISS（内存索引，每次改动从 SQLite 重建）——替代 ChromaDB，规避其异步 compaction 导致段文件不落盘的损坏风险

### 🔐 LLM 配置灵活（非绑定某一家）
- 支持任意 **OpenAI 兼容端点**（DeepSeek / 通义 / Kimi / 本地 vLLM 等）
- `base_url + api_key + 模型名` 三项可随时更改，**热更新无需重启**
- 未配置时不假装可用：右上角「⚙ API配置」弹窗引导填写，未填则明确提示

---

## 🏗️ 系统架构

```
员工提交报销单据
   │  (发票凭证 + 报销信息 + 手填条目，三者必填一体)
   ▼
[入口]  /v1/reimburse · 必填三要素校验 · 输入扫描(防注入)
   │
   ▼
[解析层]  票据识别(识图/OCR) → 结构化字段
   │
   ▼
[规则引擎]  发票查重 → 供应商黑名单 → 金额交叉核对 → 要素一致性 → 行程校验
   │
   ├─ 命中硬性规则 ──→ 直接判定 (通过/异常/需人工) · 归因 = 规则
   └─ 规则未覆盖  ──→ LLM 兜底裁决 · 归因 = 模型
   │
   ▼
[权限闸门]  模型意图 → Runtime 校验(ALLOW/CONFIRM/DENY) → 脱敏/审计/回滚
   │
   ▼
[输出]  逐条结论 + 理由 + 归因标签 + 审计记录
```

---

## 🔄 核心机制走读

### 1. 发票唯一性（恶意重复报销拦截）
系统维护已见发票集合，同一发票号二度提交即判定 `DUPLICATE_INVOICE`，拦截重复报销。

### 2. 供应商黑名单
命中黑名单供应商，直接判定 `BLACKLISTED_SUPPLIER`，不进入后续流程。

### 3. 权限闸门（Runtime 授权）
```
[模型]  ── 只输出审计意图(意图JSON)
           │
           ▼
[Runtime]  ── 七道闸门：白名单 → 身份 → 角色/Scope → 参数Schema → 预算 → 风险分级 → 审批
           │  ├─ 全部通过 → ALLOW
           │  ├─ 高危写操作 → CONFIRM（人工确认令牌，15 分钟内有效）
           │  └─ 任一不过 → DENY（默认拒绝：未注册的操作一律拒绝）
           ▼
[registry.execute]  ── 唯一执行通道，模型无直接权限
           ▼
[执行后]  ── 结果脱敏 · 审计落库 · 幂等/预算持久化 · 写操作登记回滚钩子
```
确保模型永远无法越权访问真实数据。

### 4. 防 Prompt 注入
输入经 `ContentFilter` 扫描（已知攻击模式 + 高危指令特征库），命中高危指令即隔离为 `〔标记为数据〕`，并用分界符号把用户输入与系统指令隔离。

---

## 📦 技术栈

| 层 | 技术 |
|----|------|
| 后端 | Python 3.11+ · FastAPI · Uvicorn |
| LLM | OpenAI 兼容客户端（可切换任意厂商） |
| 向量库 | **自研：SQLite（WAL 持久层）+ FAISS（内存索引）** |
| 业务库 | SQLite（知识库 / 审计记录 / 幂等与预算状态） |
| 部署 | Docker（多阶段构建）+ docker-compose |
| 前端 | 原生 HTML / CSS / JS（无框架，零依赖） |
| 文档解析 | pdf-parse / pdfjs-dist / tesseract.js / canvas |

---

## 🚀 快速开始

### 环境要求
- Python 3.11+
- 任意 OpenAI 兼容的 LLM API（用于识图兜底 / 规则漏洞裁决）

### 安装

```bash
# 1. 克隆仓库
git clone https://github.com/Jiangxi1226/reimburse-audit-platform.git
cd multimodal_rag

# 2. 安装 Python 依赖
pip install -r requirements.txt

# 3. 配置 LLM（两种方式任选）
cp .env.example .env            # 或直接在界面右上角「⚙ API配置」填写
```

> `base_url` 必填且须为 `https`；`api_key` / 模型名必填。三者均可随时更改，**热更新，无需重启**。

### 运行测试

```bash
# 回归测试（52 项：核心模块 / Agent 推理 / 权限闸门 / 冲突消解 / 流式适配 / 集成链路）
pytest tests/ -v

# 离线基准（规则引擎审计能力：allow/deny 闸门真实生效）
python reimbursement/benchmark.py
```

> 其中 3 项用例需要 `LLM_API_KEY`（校验真实 LLM 客户端初始化），未配置环境变量时会失败，属预期行为。

### 启动

```bash
# 方式一：本地启动
uvicorn api:app --port 8000

# 方式二：Docker（推荐，环境一致）
docker compose up -d
```

打开浏览器访问：[http://localhost:8000/reimburse](http://localhost:8000/reimburse)（报销审核工作台）

---

## 📡 API 参考

### 报销审核
| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/v1/reimburse` | 提交报销单据进行审核（票据 + 报销信息 + 手填条目） |
| `POST` | `/v1/reimburse/ask` | 基于单据提问 |
| `POST` | `/v1/reimburse/ask_system` | 系统级审查询问 |
| `POST` | `/v1/reimburse/ask_system/stream` | 审查询问（SSE 流式） |
| `GET` | `/v1/reimburse/records` | 查询历史审核记录 |
| `GET` | `/v1/reimburse/records/{rec_id}` | 查询单条审核记录 |
| `POST` | `/v1/reimburse/records/{rec_id}/review` | 人工复核：覆写审核结论并回流统计 |
| `GET` | `/v1/reimburse/review_stats` | 复核统计（一致率 / 规则偏严偏松 / 被推翻最多的规则） |

### 基础能力 / 知识库
| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/v1/chat` | 对话问答 |
| `POST` | `/v1/upload` | 文档入库 |
| `GET` | `/v1/upload/status/{task_id}` | 上传任务状态 |
| `POST` | `/v1/workflow` | 工作流问答 |
| `GET` | `/v1/stats` | 知识库统计 |
| `POST` | `/v1/eval` | 离线评估 |

### 系统
| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查 |
| `GET` | `/reimburse` | 报销审核前端工作台 |
| `GET` | `/setup` | 首次配置引导页 |
| `POST` | `/setup` | 保存 LLM 配置（热更新） |

---

## 🗂️ 目录结构

```
multimodal_rag/
├── api.py                 # FastAPI 服务入口（所有 HTTP 端点）
├── frontend/reimburse.html  # 报销审核 Web 界面
├── reimbursement/         # 报销业务核心
│   ├── audit_engine.py    #   规则引擎（逐条判定核心）
│   ├── invoice_parser.py  #   票据/发票解析
│   ├── service.py         #   审核服务（规则→LLM 兜底编排）
│   ├── record_store.py    #   记录存储（已见发票集合）
│   ├── rag_store.py       #   报销知识库向量检索
│   ├── reimbursement_rules.json  # 规则配置
│   └── benchmark.py       # 审计能力基准测试
├── core/                  # Agent 框架 / 权限闸门 / LLM 客户端（热更新）/ 持久化
├── rag/                   # 底层 RAG 检索 / 冲突检测 / 分块 / 精排 / 向量库
├── memory/                # 记忆系统（短期/情景/长上下文）
├── tools/                 # Agent 工具（RAG 工具/计算器/识图）
├── eval/                  # 四维评估 + 三类样本集
├── utils/                 # 安全 / 日志 / 追踪 / 装饰器
├── tests/                 # pytest 回归（52 项）
├── Dockerfile             # 多阶段构建镜像
├── docker-compose.yml     # 一键编排（数据卷挂载）
├── launcher.py            # 一键启动 + 服务管理
├── setup.html             # 首次配置引导页
├── CHANGELOG.md           # 更新日志
└── README.md
```

---

## 🧪 评估与量化

一条命令跑完全部 7 组评测：

```bash
python reimbursement/benchmark.py                     # 全量（含 OCR 与 LLM）
NO_OCR=1 NO_LLM=1 python reimbursement/benchmark.py   # 离线秒级（不烧 token）
STRICT=1 python reimbursement/benchmark.py            # 回归门禁：任一指标跌破阈值即 exit(1)
```

| 组 | 内容 | 实测结果 |
|----|------|----------|
| A | 结构化审核 17 例 | 决策 **100%**、核准金额 **100%**、逐条核定 **100%（20/20 条）** |
| A′ | 防骗保（重复发票 / 黑名单 / 申报总额不符） | 2/2，漏检 **0** |
| B | 真实票据端到端 OCR（4 张） | 处置正确率 **100%**（单张含 OCR 约 6~25s，瓶颈在 PaddleOCR） |
| C | LLM 兜底裁决 | 语义模糊点真调用并返回可解析裁决（约 1.6s） |
| D | Prompt 注入 | 基础组检出 **100%**、正常组误报 **0%**、对抗组（空格拆分 / 同义 / Unicode）**33%**（如实披露边界） |
| E | **全 LLM 对照组** | 同一批 17 例裸 LLM 直判：决策 **41.2%~47.1%**、决策+金额全对 **29.4%**、耗时 **714ms**；规则侧 **100% / 0.1ms** |
| F | 人工复核回流闭环 | 覆写结论 → 一致率统计断言通过 |

**E 组的价值**：同一份测试集跑两次，裸 LLM 分别得到 41.2% 与 47.1%，实证其**不可复现**；规则侧则是恒定 100%。这是"为什么不让大模型直接算钱"最直接的数据回答（**准确率约 1/3、耗时 714 倍**）。

**分阶段错误归因**：端到端失败时自动定位是"OCR/抽取阶段"还是"规则阶段"，并能区分「字段抽取错」「输入数据过期」「规则判定错」三种成因，不让失败结果变成一团迷雾。

关键指标与阈值集中定义在 `benchmark.py` 的 `THRESHOLDS`，`STRICT=1` 时未达标即非零退出，可直接挂 CI 作为回归门禁。

**人工复核回流**：`POST /v1/reimburse/records/{rec_id}/review` 覆写结论，`GET /v1/reimburse/review_stats` 汇总一致率 / 规则偏严 / 偏松 / 被推翻最多的规则 —— 回答"规则是写死的，判错了怎么发现、怎么改"。

RAG 问答侧另有 `eval/` 四维评估（忠实度 / 相关性 / 上下文精度 / 引用准确率）+ 幻觉检测器 + 三类针对性样本集（易误读 / 冲突 / 拒答），可用 `python eval/run_evaluation.py` 复跑；其中不使用 LLM 的维度支持 `--no-llm` 离线先验检索层。

RAG 问答侧另有 `eval/` 四维评估（忠实度 / 相关性 / 上下文精度 / 引用准确率）+ 幻觉检测器 + 三类针对性样本集（易误读 / 冲突 / 拒答），可用 `python eval/run_evaluation.py` 复跑；其中不使用 LLM 的维度支持 `--no-llm` 离线先验检索层。

---

## 🛡️ 安全设计

- **密钥不入库**：`LLM_API_KEY` 仅存于项目根 `.env`（已被 `.gitignore` 忽略），绝不提交版本库
- **CORS 收敛**：不随意外发 `*` + 凭据组合
- **异常不泄露栈**：异常详情记服务器日志，对外 `detail=None`
- **records 端点鉴权**：默认本地回环保护，配置 API_KEY 后升级为鉴权
- **注入防护**：`utils/security.py` 扫描 + 隔离 + 分界符号；配套注入评测集实测（基础组检出 100%、正常组误报 0%、对抗组 33% 如实披露）
- **审计脱敏**：入审计库前递归扫描参数全部字符串值，命中手机号 / 证件号 / 银行卡 / 金额即打码
- **资源上限**：限流器、上传任务表、权限闸门内部状态均设硬上限，防内存无限增长

---

## 🛣️ Roadmap

- [ ] 接入真鉴权（JWT / OAuth）替代本地回环保护
- [ ] 规则引擎可视化运营后台（规则可配置热更新）
- [ ] 对接电子发票查验接口（验真）
- [ ] 权限模型已有角色 / Scope / 租户字段，待补**多租户数据面隔离**（向量库与审计按租户分库）

---

## 🤝 贡献

欢迎提 Issue / PR。请遵循现有代码风格，主分支为 `main`。

## 📄 License

Released under the [MIT License](https://opensource.org/licenses/MIT).

---

<div align="center">

**❤️ 如果对你有帮助，点个 Star 支持一下！**

</div>
