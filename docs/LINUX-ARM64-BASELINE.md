# Linux arm64 基线（2026-11-16）

一次性验证记录：在 arm64 Linux 上构建 `onnxruntime` 扩展并跑完整 PHPT 套件。
**不是长期 CI 基线**，但如果后续要接 arm64 CI（Graviton / 鲲鹏 / 飞腾），这份记录是起点。

## 环境

| 项 | 值 |
|---|---|
| 宿主 | Mac mini M2 / macOS 27.2.1，QEMU 11.1.3 + HVF 原生 arm64 虚拟化 |
| Guest | Ubuntu 24.04.3 LTS arm64，kernel 6.8.0-147-generic，4 vCPU / 6 GB |
| PHP | 8.5.12，源码构建（`--disable-all --enable-cli`），NTS 与 ZTS 各一份，Module API `20250925` |
| ORT | 1.28.2 官方 `onnxruntime-linux-aarch64-1.28.2.tgz`，`ORT_API_VERSION 28` |
| 工具链 | gcc/g++ 13.3.0（Ubuntu 24.04 默认） |
| 扩展 | `git archive HEAD` 干净源码，`--with-onnxruntime=/opt/ort` |

## 结果

| 变体 | 结果 |
|---|---|
| NTS | **33 PASS / 0 FAIL / 2 SKIP / 35** |
| ZTS | **33 PASS / 0 FAIL / 2 SKIP / 35** |

两个 skip 与 Linux x86_64 完全一致：

- `025-coreml.phpt` — CoreML 仅 macOS
- `033-device-tensor-cuda.phpt` — CUDA Plugin EP 测试仅 Windows

运行时自检：`Runtime::getVersion()` = `1.28.2`，`getAvailableProviders()` = `CPUExecutionProvider`，
`getEpDevices()` = 1（内置 CPU device）。aarch64 无 CUDA 预编译包，这条基线是纯 CPU。

## 已知问题：cpuinfo 警告会污染 PHPT 输出

官方 ORT aarch64 预编译库在 x86_64 上不打印，在 aarch64 上每个进程启动时向 stderr 打印一行：

```text
onnxruntime cpuid_info warning: Unknown CPU vendor. cpuinfo_vendor value: 0
```

`run-tests.php` 合并 stdout/stderr，因此 **35 个用例全部因这一行输出不匹配而 FAIL**——不是功能失败。
用 `-p` 指向一个过滤该行的包装器后全部通过：

```bash
cat > /tmp/phpwrap <<'EOF'
#!/bin/bash
exec /opt/php/bin/php "$@" 2> >(grep -v "cpuid_info warning" 1>&2)
EOF
chmod +x /tmp/phpwrap

LD_LIBRARY_PATH=/opt/ort/lib /opt/php/bin/php run-tests.php -q -n \
  -p /tmp/phpwrap -d extension=$PWD/modules/onnxruntime.so tests/
```

结论：arm64 上跑 PHPT **必须**经过这个包装器，否则得到的是假红。x86_64 不需要。

## 复现要点

1. PHP 必须用 `--with-php-config=<prefix>/bin/php-config` 传给扩展的 `./configure`，
   否则 phpize 生成的 configure 报 `Cannot find php-config`。
2. 官方 aarch64 SDK 目录布局与 x86_64 相同（`include/` + `lib/`），`config.m4` 无需改动。
3. `-lstdc++` 分支在 arm64 上正确；`g++ 13.3` 编译无警告升级为错误。
4. 源码不在仓库里原地构建：容器/VM 中用 `git archive` 导出的副本，避免污染本机构建现场。

## 产物

完整 PHPT 输出（本地保留，未入库）：

- `logs/arm64-20261116/phpt-nts.log`
- `logs/arm64-20261116/phpt-zts.log`
