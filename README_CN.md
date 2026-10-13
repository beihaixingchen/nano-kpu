# Nano-KPU RTL 优化项目（中文版）

**语言：** [English](README.md) | 简体中文

这是 [MoonshotAI/nano-kpu](https://github.com/MoonshotAI/nano-kpu) 的个人 RTL
优化 fork。上游设计及原始归属保持不变；本 fork 发布了覆盖 7 个 RTL 模块的
V73 优化候选。

## 优化归属

在项目负责人提出目标并指导方向下，GPT-6 Luna 以 V0 为基线，独立执行了 V1 至
V73 共 73 轮 RTL 优化迭代：生成优化假设、修改 RTL、分析仿真/综合反馈，并推进或淘汰候选方案。

LLM 优化 Harness 是由 Markdown 流程说明和跟踪记录构成的文档规范，不会自动执行工具。
它要求模型为每个候选保留完整轨迹：父版本、优化假设、RTL 差异、实际命令及结果、
P0/ABC 指标、关键路径、源文件 hash，以及保留/淘汰/回退决定。GPT-6 Luna 按该规范
调用仓库 `harness/` 中的脚本，并依据结果推进迭代；这些脚本不负责生成优化 RTL。

## 优化结果

本轮探索了 70 余个 RTL 候选，并依据全模型正确性、全顶层 Nangate45 ABC 时序、
仿真 cycles/token、推导吞吐和映射逻辑面积选择 V73。

### 全程提升

| 指标 | 原始优化基线（V0） | V73 | 变化 |
|---|---:|---:|---:|
| 全顶层 Fmax（pre-layout ABC） | 128.6 MHz | 442.923 MHz | +244.4%（3.44 倍） |
| cycles/token | 11,397.083 | 12,029.271 | +5.55% |
| 推导吞吐 | 11,283.59 tok/s | 36,820.416 tok/s | +226.3%（3.26 倍） |
| 映射逻辑面积 | 1.754 mm2 | 2.050019 mm2 | +16.9%* |
| ABC gates | 1,552,365 | 1,707,772 | +10.0% |

吞吐按 `Fmax_MHz * 1,000,000 / cycles_per_token` 计算。V0 是此前已完成
DEQ/Vec 工作后的初始优化基线；下方严格功能对照使用未经修改的上游源码
commit `e1855debc4469dbfa09b18f34dfcdf9ac7b0f949`。

*V0 面积来自早期campaign报告，V73面积来自全顶层映射脚本；面积变化用于展示
整体趋势，不属于严格同流程对比。V73映射面积不含 SRAM/ROM 宏。

V73 修改了 7 个核心 RTL 文件：`msh_deq.v`、`msh_fetch.v`、`msh_gemv.v`、
`msh_kda.v`、`msh_mla.v`、`msh_seq.v` 和 `msh_vec.v`。

## 主要 RTL 改进

以下改动在 V0 到 V73 逐步累积形成，并非全部发生在最后一轮：

- **KDA 并行与重叠：**使用完整 256-bit 状态字，将处理宽度从每周期 4 行扩展到 8 行；
  让下一列 pass1 与当前列 pass2 重叠；平衡归约树，并用单一循环 one-hot bank 写索引。
- **Vector 数据通路流水：**每次 SMIX 操作只捕获一次最大值；将 SMIX 乘法器与 EW 资源共享分开；
  流水化卷积输出缩放，同时保持其发射流持续运行。
- **Fetch/Sequencer 控制：**寄存 HQ 队列非空状态，并简化描述符及控制选择路径。
- **V73 最后一轮：**将 softmax-exp 乘积移入已有 LUT 加载阶段，并把可变指数向量切片改为固定
  lane 切片；这次最后调度没有增加 cycles/token。

## 正确性验证

- Full P0 通过，覆盖 seq48 和 seq16；定点模型输出、logits/argmax、随机复位重放
  和延迟检查均通过。
- 与原始源码进行 6 个确定性场景的严格对照：3 组 seed/group-size 配置，各跑
  seq48 和 seq16。输入镜像/布局、原始输出字节、logits、argmax 和状态逐位一致；
  两版对定点模型的 logit mismatch 均为 0。
- 这 6 个场景是确定性回归证据，不是形式化等价证明。

## 指标范围

Fmax 和面积来自 Nangate45 全顶层 ABC 综合，是布局布线前估算，不是流片或后布局
布线数据。映射逻辑面积不包含 SRAM/ROM 宏。吞吐由 ABC 频率估算与仿真
cycles/token 联合推导。

## 上游项目

本仓库包含一颗由 Kimi-K3 完全自主设计与实现的 nano 规模混合架构推理芯片的
RTL 设计,以及配套的功能仿真与性能(时序/面积)测试流程。模型为
hybrid-attention MoE transformer:**KDA(Kimi Delta Attention)线性注意力**、
**NoPE 多头潜在注意力(MLA)**、**sigmoid 路由 MoE(top-2 路由专家 +
一个共享专家)**,以及 **attention-residual mixing**;int4 group-128 权重,
teacher-forced 逐 token decode。

*声明:本仓库是 Kimi K3 的能力演示(demonstration),并非 Moonshot AI 官方项目。*

## 目录结构

```
rtl/            芯片 RTL(顶层模块 msh_chip_top;filelist.f 为编译清单)
  ├── roms/     LUT 初始化 hex(sigmoid/alpha/expneg/rsqrt/recip)+ 生成器说明
  └── selfmodel/  bit-exact Python 定点模型(仿真辅助)
reference/      float32 黄金参考模型(正确性判定的基准;golden 在评测时现算)
harness/        功能仿真 + 性能评测流程
  ├── evaluate.py   主入口:正确性(cos/argmax)+ cycles/token + 吞吐
  ├── tb/           Verilator C++ testbench(16 B/cycle DRAM 端口模型)
  ├── macros/       msh_sram / msh_rom 宏模型(仿真注入 + 综合 blackbox)
  ├── synth_area.ys / synth_tech.ys / synth_netlist.ys
  │                 性能脚本:yosys 面积/NE、Nangate45 映射时序、门级网表
  ├── lib/          工艺库(setup 时下载,不随仓库分发)
  └── scoring.py / audit.py / memmap.py / check_integrity.py ...
                    scoring.py = 正确性判定 + 综合度量助手
                    (NE/latch/lint 阈值,无计分)
docs/           规格文档(architecture / interface / memory_map /
                quantization / TASK_SPEC)
weights/        权重格式说明
Makefile        统一入口
```

## 环境搭建(Setup)

克隆后先执行一次:

```bash
bash scripts/setup_env.sh      # 或:make setup
source scripts/kpu-env.sh      # 把工具链 env 加进 PATH
```

脚本以 conda 为中心,幂等:

1. **Conda**——有现成的用现成的,没有则自动装 Miniconda 到
   `~/miniconda3`(免 root)。
2. **独立 env `nano-kpu`**——从 conda-forge 创建,含 **Verilator 5.x**
   (基线 5.050)、**yosys >= 0.64**、**python 3.12 + numpy**;
   绝不动任何既有 base 环境。
3. **`scripts/kpu-env.sh`**——脚本生成的 PATH 文件;source 一下(或开新
   shell),`verilator` / `yosys` / `python3` 即指向该 env。
4. **Nangate45 工艺库**——从 OpenROAD-flow-scripts 下载
   `NangateOpenCellLibrary_typical.lib`(pin commit + SHA-256 校验)到
   `harness/lib/`。该库刻意不随仓库分发;只有综合(timing/area)需要它,
   纯仿真不需要。

前置要求:bash、`curl` 或 `wget`、可访问 repo.anaconda.com(或镜像)与
raw.githubusercontent.com。任一步失败,脚本会给出具体错误并以非零退出
——解决问题后重跑即可。

## 快速开始

依赖:Verilator 5、yosys(综合用)、python3 + numpy。

```bash
bash scripts/setup_env.sh                  # 仅首次:装工具链 + 下载工艺库
source scripts/kpu-env.sh                  # 把工具链 env 加进 PATH
python3 harness/evaluate.py --quick        # 功能仿真(短序列,分钟级)
python3 harness/evaluate.py                # 完整评测(仿真 + 时序/面积综合,小时级)
python3 harness/evaluate.py --skip-synth   # 功能 + randreset + latency,不跑综合
make synth                                 # 只跑面积/时序流程
make lint / audit / selftest               # 单项检查
```

注意:`rtl/selfmodel/run.sh` 契约使用 PATH 中的 `python3`,需带 numpy。

## 协议与测量要点(详见 docs/)

- 接口:命令流(`RUN 0x1` -> `DONE 0xD0DE`)+ 128-bit DRAM 口(读保序、
  >=24 cycle 延迟、写 posted)+ 自描述内存镜像(header + descriptor 表)。
- 正确性:对照 float32 参考,每行 logits cos >= 0.98,pooled argmax >= 0.99。
- 吞吐:`cycles_per_token = 长序列仿真拍数 / seq_len`,
  `tokens/s = 时钟 / cpt`。
- 存储规则:触发器以外的存储一律走 `msh_sram`/`msh_rom` 宏;宏 bit 按
  ~1 Mbit/mm2 计入面积预算。

**测量方法说明**:为保证测试速度,area 与 timing 在**综合(synthesis)阶段**
测得——即 yosys/abc 映射与静态时序的 pre-layout 估计,**未经过后端布局布线
(P&R)验证**。为保证不同设计之间的可比性,测量使用一套**固定、冻结的综合
参数**,刻意不为单个设计搜索最优参数;因此报告数字是保守、可复现的基线,
而非最优 QoR,也非签核(sign-off)值。

## 许可证与第三方资产

本仓库以 Apache License 2.0 发布(见 `LICENSE`)。

**Nangate45 标准单元库**(`NangateOpenCellLibrary_typical.lib`)**不随
仓库分发**,由 `scripts/setup_env.sh` 在 setup 时从
[The-OpenROAD-Project/OpenROAD-flow-scripts](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts)
(`flow/platforms/nangate45/lib/`)获取。
