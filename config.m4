PHP_ARG_WITH([onnxruntime],
  [whether to build with ONNX Runtime support],
  [AS_HELP_STRING([--with-onnxruntime[=DIR]],
    [ONNX Runtime source: DIR = SDK with headers/libs; bare option = search the system libraries; omit entirely = download the CPU package from NuGet])],
  [auto])

PHP_ARG_WITH([cuda12],
  [whether to target the CUDA 12 ONNX Runtime package],
  [AS_HELP_STRING([--with-cuda12],
    [Download the NuGet GPU package built for CUDA 12 (Linux only, requires --with-onnxruntime to be omitted)])],
  [no],
  [no])

PHP_ARG_WITH([cuda13],
  [whether to target the CUDA 13 ONNX Runtime package],
  [AS_HELP_STRING([--with-cuda13],
    [Download the NuGet GPU package built for CUDA 13 (Linux only, requires --with-onnxruntime to be omitted)])],
  [no],
  [no])

PHP_ARG_WITH([coreml],
  [whether to require the CoreML execution provider],
  [AS_HELP_STRING([--with-coreml],
    [Require CoreML support (macOS arm64 only; the CPU NuGet package already embeds it)])],
  [no],
  [no])

if test "$PHP_ONNXRUNTIME" = "no"; then
  AC_MSG_RESULT([ONNX Runtime support disabled])
else

PHP_REQUIRE_CXX()

PHP_ADD_INCLUDE([$abs_srcdir/src])

ort_source=""
ort_root=""
ort_include_dir=""
ort_lib_dir=""
ort_lib_extra=""
ort_pkg=""
ort_ver=""
ort_rid=""

case "$host_os" in
  darwin*)
    ort_os=darwin
    ;;
  linux*)
    ort_os=linux
    ;;
  *)
    AC_MSG_ERROR([ONNX Runtime builds are supported on Linux and macOS only; use config.w32 on Windows])
    ;;
esac

case "$host_cpu" in
  x86_64|amd64|x64)
    ort_arch=x64
    ;;
  aarch64|arm64)
    ort_arch=arm64
    ;;
  *)
    AC_MSG_ERROR([unsupported CPU architecture for ONNX Runtime: $host_cpu])
    ;;
esac

if test "$PHP_CUDA12" != "no" && test "$PHP_CUDA13" != "no"; then
  AC_MSG_ERROR([--with-cuda12 and --with-cuda13 are mutually exclusive])
fi

ort_wants_cuda=no
if test "$PHP_CUDA12" != "no" || test "$PHP_CUDA13" != "no"; then
  ort_wants_cuda=yes
fi

if test "$ort_wants_cuda" = "yes"; then
  if test "$ort_os" != "linux"; then
    AC_MSG_ERROR([CUDA packages are only available for Linux; macOS builds use --with-coreml instead])
  fi
  if test "$ort_arch" != "x64"; then
    AC_MSG_ERROR([NuGet has no Linux $ort_arch CUDA package; CUDA builds require linux-x64])
  fi
  if test "$PHP_ONNXRUNTIME" != "auto"; then
    AC_MSG_ERROR([--with-cuda12/--with-cuda13 select a NuGet package; do not combine them with --with-onnxruntime=DIR or a bare --with-onnxruntime])
  fi
fi

if test "$PHP_COREML" != "no"; then
  if test "$ort_os" != "darwin"; then
    AC_MSG_ERROR([CoreML is only available on macOS])
  fi
  if test "$ort_arch" != "arm64"; then
    AC_MSG_ERROR([CoreML builds require macOS arm64; no x86_64 ONNX Runtime binary is published on NuGet])
  fi
  if test "$ort_wants_cuda" = "yes"; then
    AC_MSG_ERROR([--with-coreml cannot be combined with a CUDA package])
  fi
fi

dnl ---------------------------------------------------------------- source mode
case "$PHP_ONNXRUNTIME" in
  auto)
    ort_source=nuget
    ;;
  yes)
    ort_source=system
    ;;
  *)
    ort_source=dir
    ort_root="$PHP_ONNXRUNTIME"
    ;;
esac

dnl ------------------------------------------------------------------- NuGet
if test "$ort_source" = "nuget"; then
  if test "$PHP_CUDA12" != "no"; then
    ort_pkg=microsoft.ml.onnxruntime.gpu.linux
    ort_ver="${ONNXRUNTIME_CUDA12_VERSION:-1.26.0}"
  elif test "$PHP_CUDA13" != "no"; then
    ort_pkg=microsoft.ml.onnxruntime.gpu.linux
    ort_ver="${ONNXRUNTIME_CUDA13_VERSION:-1.32.0}"
  else
    ort_pkg=microsoft.ml.onnxruntime
    ort_ver="${ONNXRUNTIME_VERSION:-1.32.0}"
  fi

  if test "$ort_os" = "darwin"; then
    ort_rid="osx-$ort_arch"
  else
    ort_rid="linux-$ort_arch"
  fi

  AC_MSG_CHECKING([ONNX Runtime NuGet package])
  AC_MSG_RESULT([$ort_pkg $ort_ver ($ort_rid)])

  dnl Keep the cache outside the source tree: `make clean` runs
  dnl `find . -name '*.so' -delete`, which would otherwise remove the NuGet
  dnl runtime library and break the next build.  third_party/nuget is a
  dnl convenience symlink for tools that look there.
  ort_nuget_root="${ONNXRUNTIME_NUGET_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/ext-onnxruntime/nuget}"
  if test -z "$ONNXRUNTIME_NUGET_DIR"; then
    mkdir -p "$ort_nuget_root" "$abs_builddir/third_party"
    if test ! -e "$abs_builddir/third_party/nuget" && test ! -L "$abs_builddir/third_party/nuget"; then
      ln -s "$ort_nuget_root" "$abs_builddir/third_party/nuget"
    fi
  fi
  ort_sdk_dir="$ort_nuget_root/$ort_pkg.$ort_ver"
  ort_include_dir="$ort_sdk_dir/include"
  ort_lib_dir="$ort_sdk_dir/lib"

  dnl NuGet dylibs do not agree on their LC_ID_DYLIB install name: 1.32 uses
  dnl @rpath/libonnxruntime.1.dylib, 1.21 uses
  dnl @rpath/libonnxruntime.1.21.0.dylib.  The file itself is always
  dnl libonnxruntime.dylib.  Create aliases for all observed forms, including
  dnl the exact install name recorded in the binary, and do it for cache hits
  dnl as well so an old cache cannot silently produce an unloadable module.
  ort_make_lib_aliases() {
    ort_alias_dir="$1"
    if test "$ort_os" = "darwin"; then
      if test -f "$ort_alias_dir/libonnxruntime.dylib"; then
        ( cd "$ort_alias_dir" && ln -sf libonnxruntime.dylib libonnxruntime.1.dylib )
        ln -sf libonnxruntime.dylib "$ort_alias_dir/libonnxruntime.$ort_ver.dylib"
        if command -v otool >/dev/null 2>&1; then
          ort_install_name=$(otool -D "$ort_alias_dir/libonnxruntime.dylib" 2>/dev/null | sed -n '2p')
          ort_install_base=$(printf '%s\n' "$ort_install_name" | sed 's|.*/||')
          case "$ort_install_base" in
            *.dylib)
              if test "$ort_install_base" != "libonnxruntime.dylib"; then
                ln -sf libonnxruntime.dylib "$ort_alias_dir/$ort_install_base"
              fi
              ;;
          esac
        fi
      fi
    fi
    if test "$ort_os" = "linux" && test -f "$ort_alias_dir/libonnxruntime.so"; then
      ( cd "$ort_alias_dir" && ln -sf libonnxruntime.so libonnxruntime.so.1 )
    fi
  }

  if test -f "$ort_include_dir/onnxruntime_cxx_api.h" && \
     (test -s "$ort_lib_dir/libonnxruntime.so" || test -s "$ort_lib_dir/libonnxruntime.dylib"); then
    AC_MSG_CHECKING([cached ONNX Runtime SDK])
    AC_MSG_RESULT([$ort_sdk_dir])
  else
    AC_MSG_CHECKING([downloading ONNX Runtime from nuget.org])
    mkdir -p "$ort_nuget_root"
    ort_tmp="$ort_sdk_dir.tmp.$$"
    rm -rf "$ort_tmp"
    mkdir -p "$ort_tmp"
    ort_url="https://api.nuget.org/v3-flatcontainer/$ort_pkg/$ort_ver/$ort_pkg.$ort_ver.nupkg"
    ort_archive="$ort_tmp/package.nupkg"
    if command -v curl >/dev/null 2>&1; then
      if ! curl -fsSL --retry 3 --connect-timeout 30 -o "$ort_archive" "$ort_url"; then
        rm -rf "$ort_tmp"
        AC_MSG_ERROR([failed to download $ort_url])
      fi
    elif command -v wget >/dev/null 2>&1; then
      if ! wget -q -O "$ort_archive" "$ort_url"; then
        rm -rf "$ort_tmp"
        AC_MSG_ERROR([failed to download $ort_url])
      fi
    else
      rm -rf "$ort_tmp"
      AC_MSG_ERROR([curl or wget is required to download the ONNX Runtime NuGet package])
    fi
    if ! command -v unzip >/dev/null 2>&1; then
      rm -rf "$ort_tmp"
      AC_MSG_ERROR([unzip is required to extract the ONNX Runtime NuGet package])
    fi
    dnl Packages use build/native/include (CPU) or
    dnl buildTransitive/native/include (GPU); unzip returns an error for a
    dnl pattern that matches nothing, so extract each group separately and
    dnl validate the result below.
    unzip -q "$ort_archive" 'build/native/include/*' -d "$ort_tmp" 2>/dev/null || true
    unzip -q "$ort_archive" 'buildTransitive/native/include/*' -d "$ort_tmp" 2>/dev/null || true
    if ! unzip -q "$ort_archive" "runtimes/$ort_rid/native/*" -d "$ort_tmp"; then
      rm -rf "$ort_tmp"
      AC_MSG_ERROR([failed to extract $ort_archive for $ort_rid])
    fi

    mkdir -p "$ort_tmp/sdk/include" "$ort_tmp/sdk/lib"
    if test -d "$ort_tmp/buildTransitive/native/include"; then
      cp "$ort_tmp/buildTransitive/native/include/"*.h "$ort_tmp/sdk/include/"
    fi
    if test -d "$ort_tmp/build/native/include"; then
      cp "$ort_tmp/build/native/include/"*.h "$ort_tmp/sdk/include/"
    fi
    cp "$ort_tmp/runtimes/$ort_rid/native/"* "$ort_tmp/sdk/lib/"
    if test ! -f "$ort_tmp/sdk/include/onnxruntime_cxx_api.h"; then
      rm -rf "$ort_tmp"
      AC_MSG_ERROR([the NuGet package did not contain ONNX Runtime headers])
    fi
    if test ! -s "$ort_tmp/sdk/lib/libonnxruntime.so" && \
       test ! -s "$ort_tmp/sdk/lib/libonnxruntime.dylib"; then
      rm -rf "$ort_tmp"
      AC_MSG_ERROR([the NuGet package contained no ONNX Runtime library for $ort_rid])
    fi

    rm -rf "$ort_sdk_dir"
    mv "$ort_tmp/sdk" "$ort_sdk_dir"
    rm -rf "$ort_tmp"
    AC_MSG_RESULT([$ort_sdk_dir])
  fi
  ort_make_lib_aliases "$ort_lib_dir"
fi

if test "$ort_source" = "nuget"; then
  ort_lib_extra="-L$ort_lib_dir"
fi

dnl ------------------------------------------------------- directory-provided
if test "$ort_source" = "dir"; then
  if test ! -d "$ort_root"; then
    AC_MSG_ERROR([ONNX Runtime SDK directory does not exist: $ort_root])
  fi
  AC_MSG_CHECKING([for ONNX Runtime headers])
  for ort_candidate in "$ort_root/include" "$ort_root"; do
    if test -f "$ort_candidate/onnxruntime_cxx_api.h"; then
      ort_include_dir="$ort_candidate"
      break
    fi
  done
  if test -z "$ort_include_dir"; then
    AC_MSG_ERROR([onnxruntime_cxx_api.h not found under $ort_root])
  fi
  AC_MSG_RESULT([$ort_include_dir])

  AC_MSG_CHECKING([for ONNX Runtime library])
  for ort_candidate_lib in "$ort_root/lib" "$ort_root"; do
    for ort_candidate_file in "$ort_candidate_lib"/libonnxruntime.so \
                             "$ort_candidate_lib"/libonnxruntime.dylib; do
      if test -f "$ort_candidate_file"; then
        ort_lib_dir="$ort_candidate_lib"
        break 2
      fi
    done
    for ort_candidate_file in "$ort_candidate_lib"/libonnxruntime.*; do
      if test -f "$ort_candidate_file"; then
        ort_lib_dir="$ort_candidate_lib"
        break 2
      fi
    done
  done
  if test -z "$ort_lib_dir"; then
    AC_MSG_ERROR([libonnxruntime shared library not found under $ort_root])
  fi
  AC_MSG_RESULT([$ort_lib_dir])
  ort_lib_extra="-L$ort_lib_dir"
fi

dnl ---------------------------------------------------------- system-provided
if test "$ort_source" = "system"; then
  AC_MSG_CHECKING([for a system ONNX Runtime installation])

  if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists onnxruntime 2>/dev/null; then
    ort_pc_inc="$(pkg-config --variable=includedir onnxruntime 2>/dev/null)"
    if test -n "$ort_pc_inc"; then
      if test -f "$ort_pc_inc/onnxruntime_cxx_api.h"; then
        ort_include_dir="$ort_pc_inc"
      elif test -f "$ort_pc_inc/onnxruntime/onnxruntime_cxx_api.h"; then
        ort_include_dir="$ort_pc_inc/onnxruntime"
      fi
    fi
    ort_pc_lib="$(pkg-config --variable=libdir onnxruntime 2>/dev/null)"
    if test -n "$ort_pc_lib"; then
      ort_lib_dir="$ort_pc_lib"
    fi
  fi

  if test -z "$ort_include_dir"; then
    for ort_candidate in /usr/include/onnxruntime /usr/local/include/onnxruntime \
                         /opt/homebrew/include/onnxruntime /usr/include /usr/local/include \
                         /opt/homebrew/include; do
      if test -f "$ort_candidate/onnxruntime_cxx_api.h"; then
        ort_include_dir="$ort_candidate"
        break
      fi
    done
  fi

  if test -z "$ort_lib_dir" || test ! -d "$ort_lib_dir"; then
    ort_lib_dir=""
    for ort_candidate_lib in "/usr/lib/$host_cpu-linux-gnu" /usr/lib64 /usr/local/lib \
                              /opt/homebrew/lib /usr/lib /lib64 /lib; do
      for ort_candidate_file in "$ort_candidate_lib"/libonnxruntime.so \
                                "$ort_candidate_lib"/libonnxruntime.dylib; do
        if test -f "$ort_candidate_file"; then
          ort_lib_dir="$ort_candidate_lib"
          break 2
        fi
      done
    done
  fi

  if test -z "$ort_include_dir" || test -z "$ort_lib_dir"; then
    AC_MSG_RESULT([not found])
    AC_MSG_ERROR([
No system ONNX Runtime development files were found.
Install libonnxruntime-dev / onnxruntime-devel, pass --with-onnxruntime=DIR,
or omit --with-onnxruntime entirely to download the NuGet package.])
  fi
  AC_MSG_RESULT([$ort_include_dir, $ort_lib_dir])

  case "$ort_lib_dir" in
    /usr/lib|/usr/lib64|/lib|/lib64|/usr/lib/*-linux-gnu)
      ;;
    *)
      ort_lib_extra="-L$ort_lib_dir"
      ;;
  esac
fi

dnl ---------------------------------------------------------------- validation
ort_api_version="$(sed -n 's/^#define[[:space:]]*ORT_API_VERSION[[:space:]]*\([0-9][0-9]*\).*/\1/p' "$ort_include_dir/onnxruntime_c_api.h" 2>/dev/null | head -n 1)"
case "$ort_api_version" in
  ''|*[!0-9]*)
    AC_MSG_ERROR([could not read ORT_API_VERSION from $ort_include_dir/onnxruntime_c_api.h])
    ;;
esac
if test "$ort_api_version" -lt 18; then
  AC_MSG_ERROR([
ONNX Runtime API version $ort_api_version is too old.
The core extension requires ONNX Runtime 1.18 (API 18) or newer.  EP-device
and DeviceTensor features additionally require 1.23 (API 23); builds against
1.18-1.22 keep the class surface but those operations throw at runtime.])
fi
AC_MSG_CHECKING([ONNX Runtime API version])
AC_MSG_RESULT([$ort_api_version])

dnl Helper scripts (build-smoke.sh/run-sanitizers.sh) use exactly the SDK this
dnl configure run selected, not whichever version happens to sort first in the
dnl NuGet cache.
printf '%s\n' "$ort_include_dir" > "$abs_builddir/.ort-sdk-include"

if test -z "$ort_ver"; then
  if test -n "$ort_root" && test -f "$ort_root/VERSION_NUMBER"; then
    ort_ver="$(cat "$ort_root/VERSION_NUMBER")"
  elif command -v pkg-config >/dev/null 2>&1 && pkg-config --exists onnxruntime 2>/dev/null; then
    ort_ver="$(pkg-config --modversion onnxruntime 2>/dev/null)"
  else
    ort_ver="unknown"
  fi
fi

PHP_ADD_INCLUDE([$ort_include_dir])
if test -n "$ort_lib_extra"; then
  ONNXRUNTIME_SHARED_LIBADD="$ort_lib_extra -lonnxruntime $ONNXRUNTIME_SHARED_LIBADD"
else
  ONNXRUNTIME_SHARED_LIBADD="-lonnxruntime $ONNXRUNTIME_SHARED_LIBADD"
fi

dnl Let the platform pick its default C++ runtime.  Hard-coding libc++ breaks
dnl Linux/glibc toolchains that link libstdc++ (and vice versa).
case $host_os in
  darwin*|freebsd*|openbsd*|netbsd*)
    PHP_ADD_LIBRARY([c++], [1], [ONNXRUNTIME_SHARED_LIBADD])
    ;;
  *)
    dnl Linux defaults to libstdc++; honour an explicit -stdlib=libc++.
    case "$CXX $CXXFLAGS $LDFLAGS" in
      *-stdlib=libc++*)
        PHP_ADD_LIBRARY([c++], [1], [ONNXRUNTIME_SHARED_LIBADD])
        ;;
      *)
        PHP_ADD_LIBRARY([stdc++], [1], [ONNXRUNTIME_SHARED_LIBADD])
        ;;
    esac
    ;;
esac

dnl Runtime search path.  Explicit ONNXRUNTIME_RPATH wins; NuGet builds point
dnl at their private SDK copy so the module loads without extra env vars.
ort_rpath="${ONNXRUNTIME_RPATH:-}"
if test -z "$ort_rpath" && test "$ort_source" = "nuget"; then
  ort_rpath="$(cd "$ort_lib_dir" && pwd)"
fi
if test -n "$ort_rpath"; then
  ONNXRUNTIME_SHARED_LIBADD="$ONNXRUNTIME_SHARED_LIBADD -Wl,-rpath,$ort_rpath"
fi
PHP_SUBST([ONNXRUNTIME_SHARED_LIBADD])

dnl --------------------------------------------- CUDA / CoreML capability checks
if test "$ort_wants_cuda" = "yes"; then
  if test "$PHP_CUDA12" != "no"; then
    ort_cuda_major=12
  else
    ort_cuda_major=13
  fi

  AC_MSG_CHECKING([for CUDA $ort_cuda_major runtime libraries])
  ort_cuda_found=no
  if command -v ldconfig >/dev/null 2>&1 && \
     ldconfig -p 2>/dev/null | grep -q "libcublas.so.$ort_cuda_major"; then
    ort_cuda_found=yes
  fi
  for ort_cuda_dir in "/usr/lib/$host_cpu-linux-gnu" /usr/lib64 /usr/local/cuda/lib64 \
                      "${CUDA_HOME:+$CUDA_HOME/lib64}" "${CUDA_PATH:+$CUDA_PATH/lib64}"; do
    for ort_cuda_file in "$ort_cuda_dir"/libcublas.so."$ort_cuda_major"*; do
      if test -f "$ort_cuda_file"; then
        ort_cuda_found=yes
        break 2
      fi
    done
  done
  if test "$ort_cuda_found" = "yes"; then
    AC_MSG_RESULT([found])
  else
    AC_MSG_RESULT([not found])
    AC_MSG_WARN([
CUDA $ort_cuda_major runtime libraries (libcublas.so.$ort_cuda_major) were not
found on this machine.  The extension will still build and package, but the
CUDA execution provider will fail to load until the matching NVIDIA runtime
and cuDNN 9 are installed on the target host.  Set ONNXRUNTIME_STRICT_CUDA_CHECK=1
to turn this warning into an error.])
    if test "${ONNXRUNTIME_STRICT_CUDA_CHECK:-0}" != "0"; then
      AC_MSG_ERROR([CUDA $ort_cuda_major runtime not found and ONNXRUNTIME_STRICT_CUDA_CHECK is set])
    fi
  fi

  AC_MSG_CHECKING([for cuDNN 9])
  ort_cudnn_found=no
  if command -v ldconfig >/dev/null 2>&1 && ldconfig -p 2>/dev/null | grep -q "libcudnn.so.9"; then
    ort_cudnn_found=yes
  fi
  for ort_cudnn_dir in "/usr/lib/$host_cpu-linux-gnu" /usr/lib64 /usr/local/cuda/lib64 \
                        "${CUDA_HOME:+$CUDA_HOME/lib64}" "${CUDA_PATH:+$CUDA_PATH/lib64}"; do
    for ort_cudnn_file in "$ort_cudnn_dir"/libcudnn.so.9*; do
      if test -f "$ort_cudnn_file"; then
        ort_cudnn_found=yes
        break 2
      fi
    done
  done
  if test "$ort_cudnn_found" = "yes"; then
    AC_MSG_RESULT([found])
  else
    AC_MSG_RESULT([not found])
    AC_MSG_WARN([cuDNN 9 (libcudnn.so.9) was not found; CUDA inference will fail until it is installed])
  fi

  if test "$PHP_CUDA12" != "no"; then
    AC_DEFINE([PHP_ORT_WITH_CUDA12], [1], [Build targets the CUDA 12 provider package])
    AC_DEFINE([PHP_ORT_CUDA_MAJOR], [12], [CUDA major version targeted by the build])
  else
    AC_DEFINE([PHP_ORT_WITH_CUDA13], [1], [Build targets the CUDA 13 provider package])
    AC_DEFINE([PHP_ORT_CUDA_MAJOR], [13], [CUDA major version targeted by the build])
  fi
fi

if test "$PHP_COREML" != "no"; then
  AC_MSG_CHECKING([for the CoreML framework])
  if test -d "/System/Library/Frameworks/CoreML.framework"; then
    AC_MSG_RESULT([found])
  else
    AC_MSG_ERROR([CoreML.framework is missing; CoreML requires macOS 10.13 or newer])
  fi
  AC_DEFINE([PHP_ORT_WITH_COREML], [1], [CoreML execution provider is required])
fi

AC_DEFINE([HAVE_ONNXRUNTIME], [1], [Define to 1 if the ONNX Runtime extension is available])
AC_DEFINE_UNQUOTED([PHP_ORT_ORT_VERSION], ["$ort_ver"], [ONNX Runtime version used for the build])
AC_DEFINE_UNQUOTED([PHP_ORT_BUILD_API_VERSION], [$ort_api_version], [ORT_API_VERSION used for the build])
AC_DEFINE_UNQUOTED([PHP_ORT_PACKAGE], ["$ort_pkg"], [NuGet package used for the build, empty for external SDKs])
AC_DEFINE_UNQUOTED([PHP_ORT_SOURCE], ["$ort_source"], [Where the ONNX Runtime SDK came from])

PHP_NEW_EXTENSION([onnxruntime],
  [src/onnxruntime.cpp src/onnx_tensor.cpp src/zval_converter.cpp src/ort_value_converter.cpp src/ort_environment.cpp src/session.cpp src/session_options.cpp src/runtime.cpp src/ep_device.cpp src/async_engine.cpp src/device_tensor.cpp src/io_binding.cpp],
  [$ext_shared],,
  [-std=c++17 -DZEND_ENABLE_STATIC_TSRMLS_CACHE=1])

fi
