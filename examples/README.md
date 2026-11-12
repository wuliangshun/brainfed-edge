# 示例输出

本目录保存交付时实际运行得到的配置、日志及汇总。所有训练输入均为 synthetic 合成数据。

| 路径 | 来源 |
| --- | --- |
| `smoke/` | `configs/smoke.yaml`，seed=42 |
| `failover/` | `configs/failover.yaml`，seed=42，节点 0 在第 3、4 轮下线 |
| `comparison.csv` | LIF/MDN × 固定周期/事件稀疏，共 3 个种子、12 次训练 |
| `comparison_summary.json` | 上述运行的均值和总体标准差 |
| `election_summary.json` | 正常、首候选失效、少数派、分区及突发延迟模拟 |

为了保持源码包简洁，这里不附模型 checkpoint。请运行 README 中的命令，在自己的 `runs/` 下生成模型和完整网络 trace。例子中的墙钟耗时仅反映验证机器的一次执行。
