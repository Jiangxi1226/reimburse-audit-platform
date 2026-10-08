# 更新日志

本文件记录本项目的所有重要变更。日期格式为 `YYYY-MM-DD`。

---

## 2026-10-08 — 评测体系扩容与回归门禁

评测从"跑一遍看结果"升级为"可断言的回归门禁"，并在扩容过程中定位并修复了 7 个真实缺陷。

### 新增

- **全 LLM 对照组（E 组）**：同一批 17 个用例除走规则引擎外，另让裸 LLM（无规则 / 无工具 / 无校验）直接判定，输出准确率与耗时对比。实测裸 LLM 决策准确率 **41.2%~47.1%**、决策+金额全对 **29.4%**、耗时 **714ms**；规则侧 **100% / 0.1ms**。两次运行结果不同，实证 LLM 不可复现。
- **Prompt 注入评测集（D 组）**：分基础组、对抗组、正常组三档，统计检出率与误报率。基础组检出 **100%**、正常组误报 **0%**、对抗组（空格拆分 / 同义词 / Unicode 变形）**33%** —— 边界如实披露，不做粉饰。
- **分阶段错误归因**：端到端失败时自动定位成因，区分「字段抽取错（OCR/抽取阶段）」「输入数据过期（越过 30 天报销窗口）」「规则判定错（规则阶段）」三类，避免失败结果无从下手。
- **逐条核定准确率**：由整单结论精确到每一笔条目（如 4 笔批对 3 笔记 75%），指标口径与"逐条判定"的真实业务模式对齐。
- **人工复核回流闭环（F 组）**：新增 `record_store.apply_review()` / `review_stats()`，记录人工覆写并统计一致率、规则偏严 / 偏松、被推翻最多的规则。
- **新增 API 端点**：
  - `POST /v1/reimburse/records/{rec_id}/review` —— 人工覆写审核结论
  - `GET /v1/reimburse/review_stats` —— 复核统计
- **回归门禁**：`benchmark.py` 新增 `THRESHOLDS` 阈值表，`STRICT=1` 时任一指标跌破阈值即 `sys.exit(1)`，可直接挂 CI。

### 修复

1. **LLM 兜底裁决从未真正执行（最严重）** —— `service.py` 中 `_LLM_ARBITRATE_PROMPT` 的 JSON 示例花括号未转义，`str.format()` 将其视作占位符抛出 `KeyError`，异常被静默吞掉后 `llm_arbitrate()` 恒返回 `{}`。项目核心卖点「LLM 兜底语义模糊」在实际运行中**一次都没有生效过**，且对调用方完全静默。修正为 `{{...}}` 转义后，兜底路径恢复真实调用。
2. **高危注入表形同虚设** —— `utils/security.py` 的 `ContentFilter.scan()` 在 `high_hit` 计算之前就用 `if not matches: return safe` 提前返回，导致只能被高危表命中的句式（如"忽略之前的所有指令"）永久漏检。基础组检出率 88% → **100%**。**项目二 `medical_assistant/utils/security.py` 存在同源缺陷，已同步修复。**
3. **兜底失败言行不一** —— `service.py` 在 LLM 兜底失败时，`summary` 提示"待人工复核"但 `decision` 仍为 `approve`（前端只读 `decision` 会直接放行存疑单据）。改为同步降级为 `manual_review` 并同步 `issue_count`。
4. **测试集时间腐化** —— 结构化用例硬编码日期，随时间推移越过 30 天报销窗口，导致 4 例本应 `approve` 的单据被误判 `manual_review`。改用相对今天的 `_d(N)` 动态生成。
5. **样例票据图片日期硬编码** —— `generate_sample_receipts.py` 将日期画死在图片上，端到端测试同样随时间腐化（B 组 50%）。改用 `_d(N)` 并重新生成图片，B 组恢复 **100%**。
6. **基准观测项恒为 0** —— `benchmark.py` 的 `fallback_fail` 检测查了 `policy_references`，而 `LLM_FALLBACK_FAIL` 实际追加在 `issues`，导致该项从未被观测到。
7. **归因盲点** —— 原归因逻辑把"输入数据过期"误报为"规则阶段错"，已增加 `PAST_WINDOW` 判别，区分测试数据问题与规则缺陷。

### 变更

- `reimbursement/benchmark.py` 重构为 A / A′ / B / C / D / E / F 七组 + 门禁，支持 `NO_OCR` / `NO_LLM` / `ALL_LLM` / `STRICT` 环境变量开关。
- `reimbursement/generate_sample_receipts.py` 票据日期改为相对生成。
- README 更新「评估与量化」「API 参考」「安全设计」章节，并补充 `CHANGELOG.md`。

### 说明

以上第 1、2 条是"看结果全绿、实则功能未生效"的典型静默失败，均由本次引入的门禁与对抗用例暴露。
