# BrainFed Edge

面向边缘智能的类脑联邦学习研究原型。包括 Sponsor 聚合节点选举、事件驱动稀疏聚合、二阶元动态神经元，以及音频、视觉和雷达任务，提供可运行、可修改的 Python/PyTorch 代码。

## 快速开始

建议 Python 3.10–3.12。验证环境使用 Python 3.12、PyTorch 2.5.1 CPU。GPU 用户请先按 [PyTorch 官方说明](https://pytorch.org/get-started/locally/)安装适合本机的版本。

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell 使用 .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .

brainfed run --config configs/smoke.yaml --output runs/smoke
brainfed evaluate --checkpoint runs/smoke/model.pt
python -m unittest discover -s tests -v
```

## 主要功能

| 模块 | 已实现内容 | 主要文件 |
| --- | --- | --- |
| SNN 模型 | 两层脉冲网络、LIF/MDN 切换、脉冲率 MSE、伪梯度、序列内 BPTT | `neurons.py`, `model.py` |
| 脉冲编码 | 发放率、延迟、相位、Burst；预编码脉冲数据接口 | `encoding.py` |
| 聚合节点选举 | Sponsor、确定性任期映射、ChangeLeaderRPC/ConfirmRPC、多数派确认 | `election.py` |
| 故障仿真 | 节点下线、链路分区、丢包、延迟、突发拥塞的控制消息模拟 | `network.py` |
| 事件驱动聚合 | 神经元行显著性 EMA、累积阈值、最短/最长聚合间隔 | `aggregation.py` |
| 稀疏更新 | 同层权重行与偏置共享掩码，样本加权聚合，版本校验 | `aggregation.py` |
| 联邦训练 | 非 IID Dirichlet 划分、故障切换、重连同步、完整实验日志 | `experiment.py` |
| 对照实验 | LIF/MDN × 固定周期/事件触发四组对照，多种子汇总 | `cli.py` |

源码位于 `src/brainfed/`。本版本不依赖外部联邦学习框架，便于逐步替换算法。

## 运行选举和训练对照

```bash
brainfed election-demo --output runs/election
brainfed run --config configs/failover.yaml --output runs/failover
brainfed compare --config configs/smoke.yaml --seeds 11 23 37 --output runs/comparison
```

## 接入三类真实数据

| 任务 | 特征维度与网络 | 任务书样本规模 | 配置 |
| --- | --- | --- | --- |
| 紧急车辆警报声 | 128–2000–3 | 3 类，共 600 样本 | `configs/audio.yaml` |
| 交通图像 | 1728–2500–3 | 3 类，共 872 样本 | `configs/vision.yaml` |
| 毫米波雷达手势 | 4800–1000–5 | 5 类，共 1695 样本 | `configs/radar.yaml` |

```bash
brainfed run --config configs/audio.yaml --output runs/audio
brainfed run --config configs/vision.yaml --output runs/vision
brainfed evaluate --checkpoint runs/vision/model.pt --rotation 15
brainfed run --config configs/radar.yaml --output runs/radar
```
配置文件中的 `data.path` 相对于该 YAML 文件所在目录解析。默认 CPU 配置可改为 `device: cuda`。原规模网络比 smoke 配置更耗时；现有测试检查了它们的前向和反向传播，未执行真实数据训练。

## 输出与可复现性

每次运行生成以下文件：

- `config.json`、`environment.json`：实际使用的配置与运行环境。
- `data_provenance.json`、`partition.json`：数据来源、真实 NPZ 的 SHA256，以及训练集的客户端划分。
- `metrics.csv`：全局测试精度、宏 F1、MSE、聚合状态、模型版本和通信载荷。
- `events.json`、`elections.json`、`network_trace.json`：聚合、恢复、选举及控制消息记录。
- `summary.json`、`model.pt`：汇总结果和用于评估的全局模型。此 checkpoint 不含完整客户端优化器与 RNG 状态，不支持中断续训。

`examples/` 如附带验证输出，其数据来源均为 synthetic，只用于核对软件运行。不同 PyTorch、设备和硬件环境的数值可能变化，CPU 同环境固定种子的重复性由测试检查。


```bash
python -m pip install -e ".[plots]"
python scripts/plot_metrics.py runs/smoke runs/failover --output runs/metrics.png
```