# Phase 3 ONNX fixtures

这些模型是提交在仓库里的确定性测试数据，PHPT 只读取本地文件，不会下载模型。

| 文件 | 用途 |
| --- | --- |
| `identity.onnx` | 单输入 `X` / 单输出 `Y`，float32 `[2,3]`，用于基本推理、错误路径和重复推理 |
| `multi_io.onnx` | 输入 `A`、`B`，输出 `sum`（Add）、`diff`（Sub），float32 `[2,3]` |
| `dynamic.onnx` | 输入 `X` / 输出 `Y`，float32 `['N',3]`；元数据中动态维度应显示为 `null` |
| `sequence_io.onnx` | 输入 `X` 是 tensor(float)，输出 `seq` 是 sequence(tensor(float))，用于验证非 Tensor 拒绝 |
| `float16_io.onnx` | 输入/输出 float16 `[2]`，用于验证模型可识别但扩展 Tensor 不支持的 dtype |
| `zero_1d.onnx` | 输入 `X` / 输出 `Y`，float32 `[0]`，用于验证零元素 Tensor |
| `zero_2d.onnx` | 输入 `X` / 输出 `Y`，float32 `[2,0]`，用于验证零元素 Tensor |
| `corrupt.onnx` | 故意损坏的字节流，用于验证模型加载错误 |

模型由 `generate.py` 生成（需要 Python `onnx` 包）；当前提交的二进制即为测试使用的版本。
