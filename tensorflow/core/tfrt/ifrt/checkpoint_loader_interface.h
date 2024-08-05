/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_CORE_TFRT_IFRT_CHECKPOINT_LOADER_INTERFACE_H_
#define TENSORFLOW_CORE_TFRT_IFRT_CHECKPOINT_LOADER_INTERFACE_H_

#include <vector>

#include "absl/status/status.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "tensorflow/core/tfrt/mlrt/bytecode/bytecode.h"
#include "tensorflow/core/tfrt/mlrt/kernel/context.h"
#include "tensorflow/core/tfrt/utils/fallback_tensor.h"

namespace tensorflow {
namespace ifrt_serving {

// This interface provides APIs for restoring a checkpoint. It is expected to be
// used at IfrtRestoreVariableOp's kernel.
class CheckpointLoaderInterface {
 public:
  virtual ~CheckpointLoaderInterface() = default;

  // Called before `Load` to do some preparation work.
  virtual absl::Status PrepareRestore(
      mlir::OwningOpRef<mlir::ModuleOp> module) = 0;
  // Load the checkpoint. This API is designed to be compatible with the
  // `tf_mlrt.ifrt_restore_variable` kernel.
  virtual absl::Status Load(
      const tensorflow::tfrt_stub::FallbackTensor& prefix,
      const std::vector<tensorflow::tfrt_stub::FallbackTensor>& var_handles,
      const tensorflow::tfrt_stub::FallbackTensor& tensor_names,
      const tensorflow::tfrt_stub::FallbackTensor& shape_and_slices,
      const mlrt::bc::Vector<tensorflow::DataType>& restored_dtypes,
      const mlrt::bc::Vector<bool>& truncate_in_cast,
      tf_mlrt::Context& context) = 0;
};

}  // namespace ifrt_serving
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_IFRT_CHECKPOINT_LOADER_INTERFACE_H_
