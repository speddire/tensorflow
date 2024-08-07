/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

include "mlir/IR/OpBase.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "mhlo/IR/hlo_ops.td"
include "mlir/IR/CommonAttrConstraints.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_op_enums.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"

def CreateTFLCastToInt32Op : NativeCodeCall<
  "CreateCastToInt32($0, $_loc, $_builder)">;

def LegalizeTranspose : Pat<(MHLO_TransposeOp $arg, $perm),
                            (TFL_TransposeOp $arg,
                            (CreateTFLCastToInt32Op (TFL_ConstOp $perm)))>;

def LowerCbrt : Pat<(MHLO_CbrtOp $opr),
  (TFL_PowOp $opr,
    (TFL_DivOp
      (Arith_ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "1.0f">),
      (Arith_ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "3.0f">),
      TFL_AF_None)),
      [(F32Tensor $opr)]>;

//===----------------------------------------------------------------------===//
// Unary op patterns.
//===----------------------------------------------------------------------===//

foreach Mapping = [[MHLO_AbsOp, TFL_AbsOp],
                   [MHLO_BitcastConvertOp, TFL_BitcastOp],
                   [MHLO_CeilOp, TFL_CeilOp],
                   [MHLO_CosineOp, TFL_CosOp],
                   [MHLO_ExpOp, TFL_ExpOp],
                   [MHLO_FloorOp, TFL_FloorOp],
                   [MHLO_ImagOp, TFL_ImagOp],
                   [MHLO_LogOp, TFL_LogOp],
                   [MHLO_LogisticOp, TFL_LogisticOp],
                   [MHLO_NegOp, TFL_NegOp],
                   [MHLO_RealOp, TFL_RealOp],
                   [MHLO_RsqrtOp, TFL_RsqrtOp],
                   [MHLO_SineOp, TFL_SinOp],
                   [MHLO_SignOp, TFL_SignOp],
                   [MHLO_SqrtOp, TFL_SqrtOp],
                   [MHLO_TanhOp, TFL_TanhOp]] in
  def : Pat<(Mapping[0] F32Tensor:$input), (Mapping[1] $input)>;

def : Pat<(MHLO_ConvertOp $input), (TFL_CastOp $input)>;

def : Pat<(MHLO_Expm1Op F32Tensor:$x),
            (TFL_SubOp
              (TFL_ExpOp $x),
              (Arith_ConstantOp
                ConstantAttr<RankedF32ElementsAttr<[]>, "1.0f">),
              TFL_AF_None)>;

def : Pat<(MHLO_IsFiniteOp F32Tensor:$x),
                       (TFL_EqualOp
                          (TFL_SubOp $x, $x, TFL_AF_None),
                          (Arith_ConstantOp
                            ConstantAttr<RankedF32ElementsAttr<[]>, "0.0f">))>;

def : Pat<(MHLO_Log1pOp F32Tensor:$x),
                      (TFL_LogOp
                        (TFL_AddOp
                          $x,
                          (Arith_ConstantOp
                            ConstantAttr<RankedF32ElementsAttr<[]>, "1.0f">),
                          TFL_AF_None))>;


