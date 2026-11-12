# 数据格式

不随代码提供真实传感数据。默认 smoke 配置使用本地生成的类别原型加噪声数据，不调用网络。

## NPZ 约定

```python
import numpy as np

# 先在原始样本级完成训练/测试划分，再提取特征。
# x 为 float32，取值 [0,1]；y 为 int64，类别从 0 连续编号。
np.savez_compressed(
    "data/audio.npz",
    x_train=x_train, y_train=y_train,
    x_test=x_test, y_test=y_test,
)
```

| 数组 | 静态特征 | 已编码时序脉冲 |
| --- | --- | --- |
| `x_train` | `[N_train, F]` | `[N_train, T, F]` |
| `x_test` | `[N_test, F]` | `[N_test, T, F]` |
| `y_train` | `[N_train]` 整数标签 | 同左 |
| `y_test` | `[N_test]` 整数标签 | 同左 |

静态特征使用 `rate`, `latency`, `phase` 或 `burst`。已编码脉冲使用 `encoding: preencoded`，必须为 0/1，时间长度须等于 `timesteps`。

特征维度：音频 128；视觉 1728；雷达 4800。视觉旋转评估假定向量由 RGB 图像 `[3,24,24]` 按 C-H-W 顺序展平；若使用其他特征，须替换旋转实现。

## 防止数据泄漏

训练与测试划分由数据准备者明确提供。建议按采集对象、车辆、场景或采集批次分组划分。来自同一原始记录的裁剪、旋转或重叠窗口应归入同一侧；标准化参数只用训练集估计。随后，程序仅对训练索引进行客户端 Dirichlet 划分。

程序校验形状、标签范围、数值范围和 NaN，不能自动判断两个数组是否源自同一原始记录。请保留样本 ID、采集分组和预处理脚本。`partition.json` 的索引以 NPZ 内 `x_train` 的顺序为准。

当前相位编码是“特征强度映射到周期内发放相位”的参考实现，并不自动提取原始音频的声学相位。延迟编码与 Burst 编码同样接收已准备的特征；原始图像、音频及雷达处理应由课题实际流水线替换。
