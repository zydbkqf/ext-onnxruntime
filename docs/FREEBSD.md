# FreeBSD

Verified baseline: FreeBSD 14.5-RELEASE-p4 amd64, PHP 8.5.12 NTS (`php85`
package), ONNX Runtime 1.30.0 (`onnxruntime` package), clang 21.1.8.

FreeBSD has no ONNX Runtime NuGet RID. The default `./configure` (NuGet mode)
therefore stops with an instructional error. Use the system package instead:

```sh
pkg install php85 onnxruntime autoconf gmake pkgconf
phpize
./configure --with-onnxruntime          # or --with-onnxruntime=/usr/local
gmake -j4
```

The FreeBSD package installs headers under `/usr/local/include/onnxruntime`
and the library as `/usr/local/lib/libonnxruntime.so.1.30.0` with the usual
`.so` / `.so.1` symlinks. Both configure forms above find that layout;
`--with-onnxruntime=/usr/local` also accepts the `include/onnxruntime`
subdirectory.

Only the CPU provider is present in the FreeBSD package. `getAvailableProviders()`
returns `CPUExecutionProvider`; CoreML, CUDA and TensorRT are not available.

## PHPT

The ORT FreeBSD build prints this warning on stderr when the runtime is
initialized:

```text
cpuid_info.cc:105 ... Unknown CPU vendor. cpuinfo_vendor value: 0
```

`make test` merges stderr into the test output, so every PHPT fails on a diff
unless the line is filtered. Create a wrapper:

```sh
cat > "$HOME/php-ort-wrap" <<'EOF'
#!/bin/sh
err=$(mktemp /tmp/ort-err.XXXXXX)
/usr/local/bin/php "$@" 2>"$err"
st=$?
grep -v -E 'cpuid_info.cc|Unknown CPU vendor|cpuinfo_vendor value' "$err" >&2
rm -f "$err"
exit $st
EOF
chmod +x "$HOME/php-ort-wrap"
```

Run the suite with the wrapper as the test PHP:

```sh
NO_INTERACTION=1 REPORT_EXIT_STATUS=1 \
  gmake test PHP_EXECUTABLE="$HOME/php-ort-wrap"
```

`PHP_EXECUTABLE` must be overridden on the make command line; setting
`TEST_PHP_EXECUTABLE` alone is not enough because the generated Makefile uses
`PHP_EXECUTABLE`.

Result on the verified baseline:

```text
Number of tests : 37   34
Tests skipped   : 3
Tests failed    : 0
Tests passed    : 34 (100.0%)
```

The three skips are `025` (CoreML, macOS-only), `033` (CUDA Plugin EP,
Windows-only) and `036` (no CUDA/TensorRT provider in the CPU package).

## Unload behaviour

Without a linker flag, FreeBSD prints one extra line at process shutdown:

```text
__cxa_thread_call_dtors: dtr ... from unloaded dso, skipping
```

It comes from the thread-local destructors that ONNX Runtime registered after
PHP unloads the extension. `config.m4` links the module with
`-Wl,-z,nodelete` on FreeBSD so the module and its ORT dependency stay loaded
until process exit. This is FreeBSD-specific; Linux and macOS keep normal
unload behaviour. No extra test filter is needed for this line once `nodelete`
is in effect.

## Limits

- Only NTS was verified. The FreeBSD ports tree does not ship a ZTS PHP package;
  ZTS would need a source build and has not been tested.
- CPU inference, async tasks, IoBinding, batch inference and the CPU
  `EpDevice`/`DeviceTensor` paths pass. No GPU/NPU provider was tested.
- ASan/LSan and the native stress binaries were not run on FreeBSD.
