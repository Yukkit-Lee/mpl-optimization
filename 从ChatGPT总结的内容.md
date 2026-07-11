项目记忆包：基于 MPC 的特征级隐私保护多方学习论文

我现在在写一篇关于 基于多方安全计算的密文联合学习优化方法研究 的论文。总体方向是参考 PEA 论文的高层框架，但不想简单复现 PEA，而是在其 feature-level private multi-party learning backbone 上做模块化改进。

1. 当前论文主线

我的论文目标是：

在多机构协同场景下，不共享原始数据，通过特征级协同训练、异构感知初始化和改进的私有优化机制，实现兼顾隐私、效率和精度的多方学习框架。

当前系统可以理解为：

原始数据
→ 模块 A：Task-Adaptive Feature Extraction
→ 本地特征数据
→ 模块 B：Consensus-Guided Global Initialization
→ 初始全局模型 θ0
→ 模块 C：Adaptive-Clipped Private Optimization
→ 最终全局模型 θT

三个模块不是完全同等创新：

模块 A：系统前端和实验支撑，借鉴 PEA 的 feature extraction 思路，但替换了具体 extractor。
模块 B：当前最明确的主创新，解决 non-IID 下 local models 知识冲突和初始化差的问题。
模块 C：私有优化扩展模块，原来尝试 trust-aware secure DPSGD，但已改为 Adaptive Clipping + Efficient MPC Noise Backend 思路。
2. 和 PEA 的关系

PEA 的核心框架大致是：

Feature Extraction
→ Local data-based global model initialization
→ Secure DPSGD on logically global dataset

PEA 里的几个关键点：

Feature extraction
PEA 先用 data-independent feature extractor，把 raw data 转换成 extracted features，然后只在 extracted features 上训练 shallow/simple model。
Local model initialization
PEA 让每个 party 先用 DPSGD 在本地数据上训练 local model，然后把 local models secret-share 后在 MPL 中做 aggregation 和 candidate accuracy evaluation。
Secure DPSGD
PEA 把 DPSGD 放入 MPL / secret-sharing 环境中安全实现，包括 secure clipping 和 secure Gaussian noise generation。
epsilon 逻辑
PEA 全文一直贯彻 epsilon。
本地初始化阶段有 (ε
1
	​

,δ
1
	​

)，全局训练阶段有 (ε
2
	​

,δ
2
	​

)，整体隐私预算随 iteration 增长而变化。
在我的论文里，epsilon 仍然重要，但当前主要放在模块 C 的 private optimization / privacy accounting 中解释。
3. 我当前的真实代码状态
模块 A：特征提取

已经完成三个数据集的 feature-level 验证：

MNIST → HOG
CIFAR-10 → DINO / DINOv2 features
IMDb → RoBERTa features

模块 A 主要验证：

extracted features 维度更低；
TFE / shallow model 在 extracted features 上比 raw data 更容易训练；
feature-level route 可以改善效率和精度。

已有 MNIST 结果示例：

明文 HOG + LR:
Train Acc 1.0000
Test Acc  0.9000

TFE HOG 浅层:
Train Acc 0.9470
Test Acc  0.8950
Train Time 约 2.8s

TFE Raw 浅层:
Train Acc 0.9180
Test Acc  0.8200
Train Time 约 7.5s

HOG 消融：

ppc=(7,7), ori=9 → feat_dim=144, test_acc=0.8950
ppc=(4,4), ori=9 → feat_dim=441, test_acc=0.9050

IMDb 早期 Word2Vec 效果很差，后来切换到 RoBERTa。
CIFAR-10 切换为 DINO / DINOv2 特征。

模块 B：Consensus-Guided Global Initialization

当前模块 B 的核心流程是：

local pretraining
→ ensemble teacher
→ KL distillation
→ distilled local models
→ avg_model / best_model
→ secure aggregation + secure candidate evaluation
→ chosen initial global model

模块 B 的目的不是隐私保护本身，而是：

解决 non-IID 条件下 local models 知识不一致、直接聚合冲突大、初始全局模型质量差、后续 private training 收敛慢的问题。

具体解释：

non-IID 下各 client 的 local model 只学习到局部数据分布；
local models 之间存在 knowledge conflict；
直接 averaging 可能得到一个谁都不像的模型；
best local model 又可能只偏向某个 client；
因此先通过 ensemble teacher 得到跨客户端软目标；
再用 KL distillation 对齐 local models；
最后再聚合 distilled local models。

当前模块 B 的密态状态：

local pretraining：明文；
ensemble teacher：明文；
KL distillation：明文；
distilled local models 的 avg 聚合：已经支持 ABY3 密态聚合；
avg / best candidate evaluation：已经支持 ABY3 密态候选评估。

所以最准确表述是：

模块 B 是“明文知识对齐 + ABY3 密态聚合与候选评估”的混合初始化流程。

需要注意：

chosen_model 不是独立 aggregation strategy；
它只是从 avg_model 和 best_model 中选出来的更优 candidate；
严格对齐 PEA Table III 时，只比较 avg 和 best。
模块 C：Adaptive-Clipped Private Optimization

模块 C 原先是 trust-aware secure DPSGD，后来因为 trust ablation 效果不明显，已经改为：

Adaptive-Clipped Private Optimization
+ Efficient MPC Noise Generation Backend

当前模块 C 的真实逻辑：

输入：模块 B 输出的 chosen initial global model θ0
数据：logically global feature dataset D_LG^feat
训练模式：plain / fixed_dp / adaptive_dp
核心机制：
  - adaptive clipping 动态更新 C_t
  - noise_backend 支持 gaussian / binomial_mpc
输出：final global model θT

当前代码中已经移除 trust 机制：

不再使用 trust_t
不再使用 sigma_min / sigma_max / reliability scalar
改为 adaptive_dp
使用 AdaptiveClippingScheduler
使用 BinomialMPCNoiseBackend

MNIST 当前模块 C 结果示例：

mode = adaptive_dp
noise_backend = binomial_mpc
gamma = 0.5
eta_c = 0.2
C_init = 0.1
epochs = 800
batch_size = 64
epsilon estimate ≈ 8.58

final_train_accuracy = 0.9310
final_test_accuracy  = 0.8960
final clipping_bound = 0.193528

训练中 C
t
	​

 会动态变化，说明 adaptive clipping 确实在生效，例如：

epoch 1:   C_t ≈ 0.110, test_acc ≈ 0.745
epoch 89:  C_t ≈ 2.190, test_acc ≈ 0.889
epoch 800: C_t ≈ 0.194, test_acc ≈ 0.896
4. 几个重要概念已经确定
4.1 epsilon 是否还重要？

重要。

但现在不能再把 epsilon 和 trust 绑定。
epsilon 应该作为模块 C 的 privacy accounting 指标：

epsilon = 隐私预算
epsilon 越小，隐私越强，通常精度越低
epsilon 越大，隐私越弱，通常精度越高

我论文中应该说明：

模块 A 主要解决 feature space 和 efficiency；
模块 B 主要解决 initialization 和 convergence；
模块 C 负责 privacy accounting；
epsilon 主要用于模块 C 中分析 privacy–utility trade-off。
4.2 PEA 中 local model 为什么用 DPSGD？

PEA 让 local model 用 DPSGD，不是为了更高精度，而是为了：

让初始化阶段本身也有 DP 保证。

因为 local model 是从私有数据训练出来的，如果用普通 SGD，local model 本身可能泄露本地样本信息。
DPSGD 的作用是保护 local model 的训练输出，使 Protocol 4 的初始化阶段满足 (ε
1
	​

,δ
1
	​

)-DP。

我的当前代码还不是这种：

PEA:
feature extraction
→ local DPSGD models
→ secure aggregation
→ global init

我当前:
feature extraction
→ ordinary local training
→ ensemble teacher + KL
→ distilled local models
→ secure aggregation/evaluation
→ global init

如果以后要更贴近 PEA，可以把模块 B 前半段的 local training 换成 local DPSGD：

feature extraction
→ local DPSGD training
→ local DP models
→ ensemble teacher + KL
→ distilled local DP models
→ secure aggregation/evaluation
→ θ0

这时蒸馏仍然有意义，因为：

DPSGD 解决 privacy；
KL distillation 解决 non-IID 下 local knowledge conflict；
二者不是同一个作用。

但 local DPSGD 会引入噪声，可能削弱蒸馏效果，需要实验验证。

4.3 Adaptive Clipping 和 Secure DPSGD / DPSGD 的关系

Adaptive Clipping 不是完整替代 DPSGD，也不是完整替代 Secure DPSGD。
它改的是 DPSGD 里的 clipping 子模块。

标准 DPSGD：

g
ˉ
	​

i
	​

=
max(1,∥g
i
	​

∥
2
	​

/C)
g
i
	​

	​


Adaptive Clipping：

g
ˉ
	​

i
	​

=
max(1,∥g
i
	​

∥
2
	​

/C
t
	​

)
g
i
	​

	​


其中 C
t
	​

 动态更新。

关系：

DPSGD = 算法层
Secure DPSGD = 把 DPSGD 放入 MPC/secret-sharing 中安全实现
Adaptive Clipping = 替换 DPSGD 中固定 clipping bound 的规则层改动
BinomialMPCNoiseBackend = 替换 DPSGD 中噪声生成后端

所以论文中建议不要说“我提出新的 secure DPSGD”，而说：

本文提出 adaptive-clipped private optimization，并将其部署到多方密态训练流程中。

5. Figure 和 Table 对齐逻辑
Figure 2

PEA Figure 2 的图注：

End-to-end comparison of different model training methods. LR-TF-Encrypted and LR-Queqiao are our proposed methods. LR-TF-Encrypted and LR-Queqiao are non-private logistic regression models trained by TF-Encrypted and Queqiao.

理解：

Figure 2 不是完整 PEA A+B+C 流程；
它是方法级 comparison；
证明 feature extraction + shallow LR + MPL/TFE 这条路线，比 raw-data end-to-end DP deep model 更合理；
LR-TF-Encrypted 是 non-private logistic regression trained by TF-Encrypted。

我如果复现 Figure 2，不应该跑完整 A+B+C。
应该比较：

raw-data deep model NP
raw-data deep model DP
feature-based LR-TFE NP
feature-based LR-TFE DP

模型替代方案已定：

MNIST: SmallCNN 替代 LeNet
CIFAR-10: ResNet18 替代 VGG16
IMDb: TextCNN 替代 SampleNet

注意：

NP = non-private
DP = differentially private
epsilon 在 DP 曲线中作为隐私预算出现
主展示预算建议 epsilon ≈ 2.07
可以补充 epsilon ∈ {1.0, 2.07, 4.0}

当前跑 SmallCNN-NP 时出现：

it=0 test_acc=0.1243
it=1 test_acc=0.8197
it=100 test_acc=0.9903

原因：

当前 it 实际是 epoch，不是 mini-batch iteration；
PEA 图中的 iteration 更接近 optimization update steps，不应直接理解成 epoch；
SmallCNN + Adam + MNIST + NP 训练本来就很快；
不能强行让曲线形状和 PEA 的 LeNet 一样。

如果要更像 PEA，应：

将横轴改为 optimizer steps，而不是 epoch；
或把图中横轴命名为 Epoch；
或将 optimizer 从 Adam 改为 SGD；
或降低学习率；
但更推荐接受当前曲线，只解释模型和训练设置不同。
Figure 3

PEA Figure 3 是 inverse square root computation approximation error。
这是 PEA 底层 secure clipping primitive 的误差分析。

我当前没有设计新的 inverse square root protocol，因此不需要做 Figure 3。
论文中可以说明本文不聚焦底层 MPC 数值近似协议。

Figure 4

PEA Figure 4 是比较 extracted features vs raw data 的 accuracy curve。
我的模块 A 已经能做：

TFE HOG vs TFE Raw
DINO features vs raw/weak baseline
RoBERTa features vs weak baseline

Figure 4 中 epsilon 仍可作为曲线上标注，用于说明 privacy budget / iteration 的关系。
但要注意：之前某些脚本中 epsilon 只是 pea_aligned 展示口径，不一定是严格 accountant。

Figure 5

PEA Figure 5 是 convergence speed comparison of initialized global models vs randomly initialized global models。
它比较的不是初始化模型本身精度，而是：

不同初始化模型进入后续 secure/private training 后的收敛速度

我现在 Figure5 应该比较：

random_init
avg_init
best_init

或者如果采用 chosen，则：

random_init vs chosen_init

但当前代码里 Figure5 跑：

random_init / avg_init / best_init

进入 TFE / private training。

MNIST 当前 Figure5 结果：

random_init final test_acc = 0.8840
avg_init    final test_acc = 0.8830
best_init   final test_acc = 0.8840

最终几乎一样，所以不能强调 final accuracy。
应重点看：

early-stage convergence；
Epoch@0.80 / Epoch@0.85 / Epoch@0.88；
Epsilon@Accuracy；
或 0~200 epoch 局部放大图。

PEA 的 Figure5 也是强调 convergence speed，而不是只看最终精度。

Table III

PEA Table III：

Accuracy comparison of global models initialized by different aggregation methods.

它比较的是 Protocol 4 / 模块 B 输出的 initialized global model accuracy，不是后续 secure DPSGD final model accuracy。

严格对齐时只比较：

Averaging Strategy = avg_model
Accuracy Strategy = best_model

不要把 chosen_model 当独立策略，因为 chosen 只是从 avg 和 best 中选一个。

我应该做：

Dataset | Distribution | Averaging Strategy | Accuracy Strategy
MNIST   | IID / Non-IID
CIFAR   | IID / Non-IID
IMDb    | IID / Non-IID

重复 5 次，报告 mean ± std。

这张表可以验证：

IID 下 avg 通常略好；
Non-IID 下 avg / best 会因数据集不同而表现不同；
我的蒸馏版本如果有效，应在 Non-IID 复杂数据集上更稳。

如果要证明“蒸馏本身有用”，还需要额外做：

without KD vs with KD

否则 Table III 只证明 aggregation strategy，不直接证明 KD 的贡献。

Table IV

PEA Table IV 比 TF-Encrypted / Queqiao / CAPC / DDGauss。
我目前不做 Queqiao、CAPC、DDGauss，因此不适合复刻原版 Table IV。

可以做自己的 TFE-only 替代表：

Plain LR
LR-TFE
fixed_dp / adaptive_dp

并报告 5 runs mean ± std。

Table V

PEA Table V 比较：

models securely trained on logically global dataset
vs
models locally trained on one subset

这张表很适合我做。
可以在 CIFAR-10 上做 3-party TFE setting：

local subset training
vs
logically global feature dataset training

证明 global training 比 local subset 更好。

6. 当前论文推荐写作定位

不要写成：

我提出了一个全新的 A+B+C 框架。

因为 PEA 已经有 feature extraction、initialization、secure DPSGD 的高层骨架。

应该写成：

本文沿用 feature-level privacy-preserving collaborative training backbone，并重点改进 initialization 与 private optimization 阶段。

推荐定位：

PEA-style backbone
+ task-adaptive extractor replacement
+ consensus-guided KD initialization
+ adaptive-clipped private optimization

三模块作用从广义上说：

模块 A

作用：

让数据可训。

解决：

raw data 高维；
直接 private training 计算和通信成本高；
高维空间中 DP 噪声更容易破坏精度。
模块 B

作用：

让训练好起步。

解决：

non-IID 导致 local models 学到局部知识；
local knowledge conflict；
direct aggregation 不稳定；
initial global model 差导致后续训练慢、浪费隐私预算。
模块 C

作用：

让最终模型安全且尽量不掉点。

解决：

最终模型有 MIA 风险；
private optimization 必须加 DP 机制；
fixed clipping 难设，可能过裁剪或放大噪声；
adaptive clipping 让 C
t
	​

 动态调整；
binomial_mpc 后端提供新的 noise generation backend。

一句话：

三个模块依次解决“原始数据难训、异构初始化难、最终私有训练效用损失大”三个问题，共同服务于在多方场景下训练一个既保护隐私、又兼顾效率和精度的全局模型。

7. 论文摘要草稿方向

可以写成：

多方学习能够在不共享原始数据的条件下协同训练模型，但仍面临隐私、效率和精度之间的权衡。直接在高维 raw inputs 上进行 DP/MPL 训练会造成巨大的计算开销和显著精度损失；同时，non-IID 数据导致 local models 冲突，使全局初始化质量差；最终私有训练中的固定裁剪规则也会进一步损害模型效用。本文提出一个 feature-level privacy-preserving collaborative training framework，包括 task-adaptive feature extraction、consensus-guided global initialization 和 adaptive-clipped private optimization。模块 A 将 raw data 映射到紧凑特征空间，模块 B 通过 teacher ensemble 和 KL distillation 对齐异构 local knowledge 并进行密态候选聚合评估，模块 C 在 logically global feature dataset 上执行 adaptive-clipped private optimization。实验在 MNIST、CIFAR-10 和 IMDb 上验证该框架在显式 privacy budget accounting 下的 accuracy、convergence 和 efficiency。

8. 当前最好继续做的实验优先级

优先级建议：

Table III 风格实验
模块 B 初始化策略：avg vs best，IID/Non-IID，5 runs mean ± std。
模块 B KD 消融
with KD vs without KD，尤其看 Non-IID CIFAR / IMDb。
Adaptive Clipping 消融
fixed clipping vs adaptive clipping，在同一 chosen init、同一 noise backend 下比较：
final acc
convergence
epsilon
C
t
	​

 trajectory
Figure5 收敛速度图
不看最终精度，看 Epoch@Accuracy 或 Epsilon@Accuracy。
Figure2 方法级比较
SmallCNN / ResNet18 / TextCNN 的 NP/DP 与 LR-TFE NP/DP 对比。
9. 已用 / 计划使用的关键代码文件

当前主要文件：

mnist_new.py
cifar10_dino.py
imdb_RoBERTa.py

feature_kd_init.py
feature_kd_bridge.py
secure_aggregation.py

module_c_adaptive_bridge.py
module_c_adaptive_dpsgd.py

exp_figure2_comparison_gpu.py

模块 C 当前配置示例：

mode = "adaptive_dp"
noise_backend = "binomial_mpc"
gamma = 0.5
eta_c = 0.2
C_init = 0.1
noise_multiplier = 1.0
global_epochs = 800
batch_size = 64
delta = 1e-5

Figure2 对比模型配置中：

MNIST: SmallCNN
CIFAR: ResNet18
IMDb: TextCNN

NP: ordinary training
DP: DPSGD / Opacus if available
target_epsilon = 2.07
delta = 1e-5

注意：exp_figure2_comparison_gpu.py 当前记录的 it 实际是 epoch，不是 optimizer step。如果画图要对齐 PEA 的 iteration，最好改成按 mini-batch update steps 记录。

10. 后续写作时必须避免的错误
不要说模块 A 是我首创，因为 PEA 已有 feature extraction 思路。
不要说我提出 secure DPSGD，因为 PEA 已有 secure DPSGD。
不要说 chosen_model 是新的 aggregation strategy，它只是 avg/best 的选择结果。
不要把 PEA Figure 2 理解成完整 A+B+C final model 对比。
不要把 PEA 的 iteration 未经说明直接等同于 epoch。
不要说模块 B 全流程都在密态中；目前是明文 KD + ABY3 聚合评估。
不要把 trust-aware 作为当前模块 C 主线；当前已切换为 adaptive clipping。
不要在没有实验支撑时说 adaptive clipping 明显优于 fixed clipping，需要补消融。
不要把 epsilon 去掉；epsilon 仍是 private training 的核心指标。
不要把模块 C 的 binomial_mpc 后端说成真正完整实现了论文中的 MPC 协议；当前更准确是功能等价模拟 / 可插拔 backend prototype。
11. 最终推荐论文贡献点表述

推荐 contribution list：

Task-adaptive feature-level training front-end
通过 HOG / DINO / RoBERTa 将 raw inputs 转换为紧凑特征空间，降低后续私有协同训练复杂度。
Consensus-guided global initialization
针对 non-IID 场景，提出 teacher ensemble + KL distillation 的 local knowledge alignment 方法，并结合 ABY3 密态聚合与候选评估得到更可靠的初始全局模型。
Adaptive-clipped private optimization
在 logically global feature dataset 上进行最终私有训练，用 adaptive clipping 替代固定 clipping，并引入可插拔 noise backend，包括 binomial_mpc，用于改善 private optimization 的效用稳定性。
Comprehensive evaluation
在 MNIST、CIFAR-10、IMDb 上评估 feature route、initialization quality、convergence speed、privacy budget 和 private optimization 效果。
12. 一句话总括

我的论文不是从零发明 PEA 式 MPL 框架，而是在 PEA-style feature-level private training backbone 上，围绕“高维数据难训、non-IID 初始化难、固定私有优化规则损失大”三个问题，分别设计 task-adaptive feature extraction、consensus-guided KD initialization 和 adaptive-clipped private optimization，并通过 MNIST / CIFAR-10 / IMDb 实验证明其在 accuracy、convergence、efficiency 和 privacy-budget accounting 下的有效性。