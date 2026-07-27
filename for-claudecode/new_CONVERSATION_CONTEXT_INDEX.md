# Claude Code 对话上下文索引

## 1. 推荐阅读顺序

接手项目时按以下顺序阅读：

1. `plan/0727_new_cross_stage_incremental_full_feature_plan.md`
2. `for-claudecode/new_CLAUDECODE_HANDOFF.md`
3. `for-claudecode/new_CURRENT_AUTODL_WINDOW_SUMMARY.md`
4. `for-claudecode/new_LEGACY_INTERNAL_SERVER_CONVERSATION_SUMMARY.md`
5. `progressLog/project_progress.md`
6. `migration/CODEX_CONTEXT_HANDOFF.md`
7. `AUTODL_MIGRATION.md`

前三份说明“现在应该做什么”，后几份解释“项目为什么变成现在这样”。

## 2. 指令优先级与历史覆盖关系

如果历史文档之间存在冲突，按以下优先级处理：

```text
用户最新明确指令
> 0727 incremental full-feature计划
> Claude Code详细交接
> 当前AutoDL窗口摘要
> project_progress最后一条更正
> 旧服务器迁移交接
> 更早实验记录
```

已知被覆盖的历史指令：

| 旧指令/旧状态 | 当前有效状态 |
|---|---|
| CPU固定单核/单线程、`num_workers=0` | 已被用户取消；使用最大合理性能，但避免线程/进程爆炸 |
| DP-MTT是正式主线 | 已废弃并明确排除 |
| Stage C尚未开始 | 已完成54/54正式网格 |
| 主图双方使用Fixed DP-SGD | 已被用户更正为双方Adaptive DP-SGD |
| 需要补跑Ours Fixed | 已取消；优先复用既有Ours Adaptive |
| 复现CryptGPU | 当前增量计划不要求，只做有边界文字背景 |
| 重跑Stage A/Stage B | 禁止无理由重跑，优先复用 |

## 3. 当前唯一有效的主比较

```text
Full Data Feature Model:
Full features + Secure Average + Adaptive DP-SGD

Ours:
Official MTT compact features + Secure Consensus + Adaptive DP-SGD
```

Adaptive不是两方法之间的变量。双方必须对齐Adaptive clipping配置、目标
epsilon/delta、optimizer、batch、steps和evaluation schedule。

已有Ours Adaptive 3 datasets × 3 seeds优先复用。新增的正式计算原则上只有：

```text
9个Full-Data Secure-Average initializer
9个Full-Data Adaptive Stage-C run
```

## 4. 当前硬阻塞

最后一次硬件检查：

```text
GPU: No devices found
CPU cgroup: 50000 100000，约0.5 core
nproc: 1
```

与计划的RTX 5090 + 25 vCPU不一致。在硬件恢复前只能做静态实现、审计和
测试准备，不得启动pilot/formal。

## 5. 关于私有原始对话

私有历史JSONL仅用于必要审计，可能包含令牌、凭据和敏感工具输出：

```text
/root/autodl-tmp/private_codex_context/
codex_conversation_export_20260726_1522/
conversation_raw_PRIVATE.jsonl
```

规则：

1. 禁止整文件读取；
2. 只有现有交接无法回答具体历史问题时才使用`rg`关键词定位；
3. 禁止输出或提交任何令牌、密钥、认证头、代理认证或敏感原文；
4. 禁止把JSONL加入Git；
5. 本轮对话摘要完全依据安全交接文档和进度日志生成，没有读取该JSONL。

## 6. 当前Git与文档责任

分支：

```text
new-secure-feature-pipeline
```

任何修改都必须：

1. 更新`progressLog/project_progress.md`；
2. `git diff --check`；
3. 分阶段commit；
4. push GitHub；
5. 验证本地HEAD与远端分支一致。

长任务必须使用tmux/持久队列、heartbeat、status、checkpoint和resume。
