PHP_ARG_WITH([onnxruntime],
  [whether to build with ONNX Runtime support],
  [AS_HELP_STRING([--with-onnxruntime=DIR],
    [Path to the ONNX Runtime SDK root containing include/ and lib/])],
  [no])

if test "$PHP_ONNXRUNTIME" = "no"; then
  AC_MSG_ERROR([
--with-onnxruntime=DIR is required.
Point DIR at the extracted ONNX Runtime prebuilt package for this platform.
The directory must contain include/onnxruntime_cxx_api.h and lib/libonnxruntime.*.
])
fi

if test "$PHP_ONNXRUNTIME" = "yes"; then
  AC_MSG_ERROR([--with-onnxruntime requires a directory, for example --with-onnxruntime=/opt/onnxruntime-linux-x64-1.28.2])
fi

PHP_REQUIRE_CXX()

PHP_ADD_INCLUDE([$abs_srcdir/src])

ort_root="$PHP_ONNXRUNTIME"
ort_include_dir="$ort_root/include"
ort_lib_dir="$ort_root/lib"

if test ! -d "$ort_root"; then
  AC_MSG_ERROR([ONNX Runtime SDK directory does not exist: $ort_root])
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
dnl install, or set DYLD_LIBRARY_PATH/LD_LIBRARY_PATH at runtime.
ort_rpath="${ONNXRUNTIME_RPATH:-}"
if test -n "$ort_rpath"; then
  ONNXRUNTIME_SHARED_LIBADD="$ONNXRUNTIME_SHARED_LIBADD -Wl,-rpath,$ort_rpath"
fi
PHP_SUBST([ONNXRUNTIME_SHARED_LIBADD])

PHP_NEW_EXTENSION([onnxruntime],
  [src/onnxruntime.cpp src/onnx_tensor.cpp src/zval_converter.cpp src/ort_value_converter.cpp src/ort_environment.cpp src/session.cpp src/session_options.cpp src/runtime.cpp src/ep_device.cpp src/async_engine.cpp src/device_tensor.cpp src/io_binding.cpp],
  [$ext_shared],,
  [-std=c++17 -DZEND_ENABLE_STATIC_TSRMLS_CACHE=1])
