# Adaptive-only独立调参与Figure 6重建计划（2026-08-09）

## 目标

固定现有Fixed-DP三seed正式结果及其全部参数，不重跑、不调节Fixed；仅要求
Fixed与Adaptive共享CIFAR-10数据、C5 initializer、development-validation边界、
epsilon=3和delta=1e-5。根据用户与导师确认，Adaptive允许独立选择steps、batch、
learning rate、C0、gamma、eta及私有count更新日程，以最大化自身效果。

## Phase 0：保护与冻结

- 当前分支必须是`unified-experiments`，代码先提交推送再运行；
- 不纳入用户正在修改的Figure 2、master JSON、`goodPaper/`或`spu.log`；
- Fixed冻结为steps=320、LR=0.001、C=2.0、seeds=42/43/44，并逐项从旧正式
  JSON核验；
- test和locked holdout禁止读取；epsilon只表示conditional Stage-C；
- 所有实验使用tmux，记录精确命令和资源。

## Phase 1：Adaptive-only development screen

只用screen RNG=17，按冻结顺序完成三个阶段：

1. 独立搜索steps×LR×expected batch（112项）；
2. 在阶段1最佳点搜索C0×gamma×eta（240项）；
3. 在阶段2最佳点搜索warmup×update interval×count noise（48项）。

共400个Adaptive候选。每个候选都根据自己的steps、sample rate和count releases
重新使用Opacus RDP accountant校准到epsilon=3。选择规则固定为final development
accuracy最高，其次early AUC，再其次runtime较低。全部负候选保留。

## Phase 2：三seed正式复核

- 把Phase 1唯一最佳Adaptive配置及screen结果hash写入formal freeze并先提交；
- 仅运行Adaptive seeds 42/43/44；
- Fixed三seed直接逐条复制旧正式证据，不重新计算；
- 输出逐seed增益、均值/样本标准差、early AUC、C轨迹、count releases及epsilon；
- 无论结果正负都停止，不做结果后扩网格。

## Phase 3：Figure 6

- 新独立脚本：`new_scripts/new_paper_figures/new_build_figure6.py`；
- 中文注释说明运行方式、输入输出、可调视觉参数与不得篡改的实验语义；
- 三面板分别显示development accuracy、C轨迹和clipped fraction；
- Fixed蓝、Adaptive红，n=3使用真实样本标准差I型误差条；
- 标题明确“independently tuned / same epsilon”，不再写matched hyperparameters；
- 旧matched Figure 6保留为历史文件，不覆盖。

## 完成条件

screen完整、formal Adaptive 3/3完成、Fixed来源逐项不变、epsilon全部等于3、无
test/holdout读取、新Figure 6 PNG/PDF可打开、报告/进度更新、Git提交推送完成。
