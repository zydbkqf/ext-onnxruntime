#ifndef PHP_ORT_ENVIRONMENT_H
#define PHP_ORT_ENVIRONMENT_H

#include <memory>

#include <onnxruntime_cxx_api.h>

namespace php_ort {

/**
 * Process-level ONNX Runtime environment.
 *
 * MINIT creates it once; MSHUTDOWN releases the process-wide reference after
 * all normal PHP request objects (including Sessions) have been released.
 * Every Session copies a shared_ptr to the same Env, so it is never rebuilt
 * per inference call and cannot outlive its Env.
 */
void initializeEnvironment();
void shutdownEnvironment();

/**
 * Returns shared ownership of the process Env; throws ConversionError if
 * MINIT did not run.  Session keeps this shared_ptr so an Env cannot be
 * destroyed while an Ort::Session (or its ortvalues) still depends on it.
 */
std::shared_ptr<Ort::Env> environment();

}  // namespace php_ort

#endif  // PHP_ORT_ENVIRONMENT_H
