<div align="center">

# 🧾 财务报销审核平台

**规则引擎优先 + LLM 兜底的智能报销审核系统** · 逐条核验 · 风险拦截 · 全链路可解释可审计

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11+-ffd43b?style=flat-square&logo=python&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white">
  <img alt="ChromaDB" src="https://img.shields.io/badge/ChromaDB-vector%20store-9b59b6?style=flat-square">
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
1. **规则优先**：机械性、确定性的判定全部由规则引擎完成，`0% 依赖 LLM`;
2. **逐条可解释**：每个条目单独给出「通过 / 异常 / 需人工」结论 + 理由 + **归因标签**（规则 or 模型）；
3. **安全可控**：模型只输出审计**意图**，实际数据访问与执行权归 Runtime，经多道闸门校验后才放行。

---

## ✨ 核心特性

### 🔍 规则优先、LLM 兜底
| 机制 | 说明 |
|------|------|
| **规则引擎驱动** | 发票唯一性（重复报销拦截）、供应商黑名单、金额交叉核对、要素完整性与一致性、行程逻辑校验等全部由确定性规则判定 |
| **逐条判定** | 对每个条目分别给出「通过 / 异常 / 需人工」结论与理由，而非一个笼统结论 |
| **LLM 仅兜底** | 仅当规则无法覆盖（如解释性例外、模糊描述）才调用大模型，结果单独标注来源 |

### 🛡️ 可控与可审计
- **权限闸门**：模型只输出审计意图，执行权归 Runtime，多道闸门校验后放行（`ALLOW / CONFIRM / DENY` 三档）
- **脱敏与审计**：敏感字段脱敏、操作全程审计留痕、幂等与回滚保障
- **防 Prompt 注入**：输入扫描 + 高危指令隔离 + 分界符号，阻断"把模型当后门"的注入攻击

### 📄 多模态数据底座
- 内置多模态 RAG 底层：多格式文档解析（PDF / Word / Excel / PPT / 图片 / OCR）、向量检索、冲突检测
- **票据图片识图解析**：提取发票号、金额、供应商、日期等结构化字段
- ChromaDB 向量库 + SQLite 业务库（知识库 / 审计记录）

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
[Runtime]  ── 七道闸门校验：鉴权 / 权限 / 范围 / 脱敏 / 幂等 / 回滚 / 审计
           │  ├─ 放行 → ALLOW
           │  ├─ 需确认 → CONFIRM
           │  └─ 拒绝 → DENY
           ▼
[registry.execute]  ── 唯一执行通道，模型无直接权限
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
| 向量库 | ChromaDB |
| 业务库 | SQLite（知识库 / 审计记录） |
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

### 启动

```bash
# 启动 API 服务
uvicorn api:app --port 8000
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
├── core/                  # Agent 框架 + LLM 客户端（热更新）
├── rag/                   # 底层 RAG 检索 / 冲突检测 / 分块 / 精排
├── memory/                # 记忆系统（短期/情景/长上下文）
├── tools/                 # Agent 工具（RAG 工具/计算器/识图）
├── eval/                  # 四维评估 + 三类样本集
├── utils/                 # 安全 / 日志 / 追踪 / 装饰器
├── launcher.py            # 一键启动 + 服务管理
├── setup.html             # 首次配置引导页
└── README.md
```

---

## 🧪 评估与量化

| 维度 | 口径 | 数值 |
|------|------|------|
| 规则判定占比 | 全部结论由确定性规则得出 | **~100%** |
| LLM 兜底占比 | 仅规则无法覆盖时 | 极低（仅兜底） |
| 发票查重 | 重复发票拦截 | 命中即拒 |
| 供应商黑名单 | 命中即拒 | 审计能力 |
| 双通道审计 | 规则 vs LLM 归因 | 可追踪到每一分钱 |

配套 `reimbursement/benchmark.py` 离线基准测试，可验证规则闸门真实生效（`allow=10 / deny=3` 等用例）。

---

## 🛡️ 安全设计

- **密钥不入库**：`LLM_API_KEY` 仅存于项目根 `.env`（已被 `.gitignore` 忽略），绝不提交版本库
- **CORS 收敛**：不随意外发 `*` + 凭据组合
- **异常不泄露栈**：异常详情记服务器日志，对外 `detail=None`
- **records 端点鉴权**：默认本地回环保护，配置 API_KEY 后升级为鉴权
- **注入防护**：`utils/security.py` 扫描 + 隔离 + 分界符号

---

## 🛣️ Roadmap

- [ ] 接入真鉴权（JWT / OAuth）替代本地回环保护
- [ ] 规则引擎可视化运营后台（规则可配置热更新）
- [ ] 对接电子发票查验接口（验真）
- [ ] 多租户隔离与角色分级

---

## 🤝 贡献

欢迎提 Issue / PR。请遵循现有代码风格，主分支为 `main`。

## 📄 License

Released under the [MIT License](https://opensource.org/licenses/MIT).

---

<div align="center">

**❤️ 如果对你有帮助，点个 Star 支持一下！**

</div>
