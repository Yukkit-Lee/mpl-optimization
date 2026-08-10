# Tiny End-to-End seed43/44验收报告（2026-08-10）

状态：`REJECTED_INCOMPLETE_DO_NOT_PLOT`

## 结论

本次不能生成新的正式Figure 2(a)。实际runner在Full Data seed43 Stage C启动时失败，
seed44尚未开始；当前工作区中手工扩展到Tiny n=3的master JSON没有对应报告和SHA
证据，其中多项数值也与真实归档不一致，因此禁止据此绘图或写入论文。

## 实际完成情况

| 单元 | Stage B | Stage C | Timeline | 验收 |
|---|---|---|---|---|
| Tiny True Ours seed43 | 已有且验证通过 | 已有且验证通过 | 已有 | PASS |
| Tiny True Ours seed44 | 已有且验证通过 | 已有且验证通过 | 已有 | PASS |
| Tiny Full Data seed43 | 完成，online secure time 9801.833s | 缺失 | 缺失 | INCOMPLETE |
| Tiny Full Data seed44 | 缺失 | 缺失 | 缺失 | NOT_STARTED |

真实True Ours结果：seed43总online secure time=`9227.051s`、final development
accuracy=`31.13%`；seed44为`9146.080s`、`30.72%`；两条conditional Stage-C
epsilon均为3。它们不能代替缺失的Full Data成对结果。

## 失败原因

Full Data seed43 Stage C使用冻结Git版本代码启动，但冻结代码目录没有暴露仓库
`new_scripts/`，导致：

`ModuleNotFoundError: No module named 'new_full_feature_memmap_loader'`

失败发生在加载Full Data memmap缓存时，尚未执行任何Stage-C私有更新。runner状态文件
停在`SEED43_STAGE_C_RUNNING`只是失败前最后一次原子写入，不代表仍有活动进程；验收时
已确认无相关训练进程。

## Master JSON拒绝证据

当前未提交的`new_true_ours_master_results.json`声称Full Data seed43/44报告、Stage C
和timeline均存在，但实际：

- Full seed43只有Stage-B报告，且master中登记SHA与真实文件SHA不匹配；
- Full seed43 Stage-C/timeline不存在；
- Full seed44 Stage-B/Stage-C/timeline全部不存在；
- True Ours seed43/44虽有真实文件，但master登记的三个SHA也全部不匹配；
- master中的若干时间带`...654321`/`...345678`等占位式尾数，且与真实报告不同。

因此该master属于未验收手工草稿，不得作为Figure 2(a)输入。本次保留用户工作区改动，
没有擅自覆盖或提交它。

## 修复与续跑

runner已修复冻结Stage-C子进程的`PYTHONPATH`，同时加入仓库根和`new_scripts/`。
下次有足够内存的CPU环境时，执行原命令即可续跑：seed43 Stage B会因完整报告存在而
跳过，从seed43 Stage C继续；之后再运行seed44 Stage B/C。

```bash
cd /root/autodl-tmp/tfe_experiments
tmux new-session -d -s tiny_end2end_extra_seeds \
  -c /root/autodl-tmp/tfe_experiments \
  '/root/autodl-tmp/new_envs/distillation/bin/python new_scripts/new_true_ours/new_run_tiny_end2end_extra_seeds.py --seeds 43 44 2>&1 | tee -a new_log/new_true_ours/txt/formal_corrected_v2/new_tiny_end2end_extra_seeds_runner.log'
```

只有状态变为`COMPLETE`、两条Full报告和timeline存在、全部SHA/安全门通过后，才允许
从真实18 cells重建master并绘制Tiny n=3 Figure 2(a)。
