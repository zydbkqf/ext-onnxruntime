PHP_ARG_ENABLE([onnxruntime],
  [whether to enable onnxruntime support],
  [AS_HELP_STRING([--enable-onnxruntime],
    [Enable the ONNX Runtime PHP extension])],
  [yes])

if test "$PHP_ONNXRUNTIME" != "no"; then
  PHP_REQUIRE_CXX()

  PHP_ADD_INCLUDE([$abs_srcdir/src])

  dnl ORT is an external dependency.  Point the build at an installed SDK:
  dnl   ONNXRUNTIME_INCLUDEDIR=/opt/ort/include
  dnl   ONNXRUNTIME_LIBDIR=/opt/ort/lib
  dnl `ONNXRUNTIME_ROOT` is a convenience shorthand for both directories.
  ort_include_dir="${ONNXRUNTIME_INCLUDEDIR:-}"
  ort_lib_dir="${ONNXRUNTIME_LIBDIR:-}"
  if test -z "$ort_include_dir" -a -n "${ONNXRUNTIME_ROOT:-}"; then
    ort_include_dir="$ONNXRUNTIME_ROOT/include"
  fi
  if test -z "$ort_lib_dir" -a -n "${ONNXRUNTIME_ROOT:-}"; then
    ort_lib_dir="$ONNXRUNTIME_ROOT/lib"
  fi

  dnl The repository contains a macOS arm64 SDK only for local development.
  dnl Never select it implicitly: a production build must name its ORT SDK.
  if test -z "$ort_include_dir" || test -z "$ort_lib_dir"; then
    if test "${ONNXRUNTIME_USE_VENDORED:-no}" = "yes"; then
      AC_MSG_WARN([using repository-local ONNX Runtime SDK; this is development-only])
      ort_include_dir="$abs_srcdir/third_party/onnxruntime/include"
      ort_lib_dir="$abs_srcdir/third_party/onnxruntime/lib"
    else
      AC_MSG_ERROR([
ONNX Runtime SDK directories are required.
Set ONNXRUNTIME_INCLUDEDIR and ONNXRUNTIME_LIBDIR (or ONNXRUNTIME_ROOT).
For the repository's development-only macOS arm64 SDK:
  ONNXRUNTIME_USE_VENDORED=yes ./configure --enable-onnxruntime
])
    fi
  fi

  AC_MSG_CHECKING([for ONNX Runtime headers])
  if test ! -f "$ort_include_dir/onnxruntime_cxx_api.h"; then
    AC_MSG_ERROR([onnxruntime_cxx_api.h not found in $ort_include_dir])
  fi
  AC_MSG_RESULT([$ort_include_dir])

  AC_MSG_CHECKING([for ONNX Runtime library])
  if test ! -d "$ort_lib_dir"; then
    AC_MSG_ERROR([ONNX Runtime library directory does not exist: $ort_lib_dir])
  fi
  ort_lib_found=no
  for ort_candidate in "$ort_lib_dir"/libonnxruntime.*; do
    if test -f "$ort_candidate"; then
      ort_lib_found=yes
      break
    fi
  done
  if test "$ort_lib_found" != "yes"; then
    AC_MSG_ERROR([libonnxruntime shared library not found in $ort_lib_dir])
  fi
  AC_MSG_RESULT([$ort_lib_dir])

  PHP_ADD_INCLUDE([$ort_include_dir])
  dnl Add -L/-l directly.  PHP_ADD_LIBRARY_WITH_PATH also injects an absolute
  dnl rpath, which would bake the build machine's SDK path into the module.
  ONNXRUNTIME_SHARED_LIBADD="-L$ort_lib_dir -lonnxruntime $ONNXRUNTIME_SHARED_LIBADD"

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

  dnl Runtime search path is opt-in.  Use ONNXRUNTIME_RPATH for a relocatable
  dnl install, or set DYLD_LIBRARY_PATH/LD_LIBRARY_PATH at runtime.  The
  dnl development vendored SDK gets a relative rpath only when explicitly
  dnl selected above.
  ort_rpath="${ONNXRUNTIME_RPATH:-}"
  if test -n "$ort_rpath"; then
    ONNXRUNTIME_SHARED_LIBADD="$ONNXRUNTIME_SHARED_LIBADD -Wl,-rpath,$ort_rpath"
  elif test "${ONNXRUNTIME_USE_VENDORED:-no}" = "yes"; then
    case $host_os in
      darwin*)
        ONNXRUNTIME_SHARED_LIBADD="$ONNXRUNTIME_SHARED_LIBADD -Wl,-rpath,@loader_path/../third_party/onnxruntime/lib"
        ;;
    esac
  fi
  PHP_SUBST([ONNXRUNTIME_SHARED_LIBADD])

  PHP_NEW_EXTENSION([onnxruntime],
    [src/onnxruntime.cpp src/onnx_tensor.cpp src/zval_converter.cpp src/ort_value_converter.cpp src/ort_environment.cpp src/session.cpp src/session_options.cpp src/runtime.cpp src/ep_device.cpp src/async_engine.cpp src/device_tensor.cpp src/io_binding.cpp],
    [$ext_shared],,
    [-std=c++17 -DZEND_ENABLE_STATIC_TSRMLS_CACHE=1])
fi
