/* Copyright 2024 The OpenXLA Authors.

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

#include "mlir/IR/OpDefinition.h"  // IWYU pragma: keep
#include "xla/service/gpu/fusions/triton/xla_triton_ops.h"
#include "triton/Dialect/Triton/IR/Utility.h"
// #include
// "third_party/llvm/llvm-project/mlir/include/mlir/IR/OpImplementation.h"  //
// IWYU pragma: keep
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"
#include "third_party/triton/include/triton/Dialect/TritonGPU/IR/Attributes.h"  // IWYU pragma: keep
#include "third_party/triton/include/triton/Dialect/TritonGPU/IR/Dialect.h"
#include "third_party/triton/include/triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "third_party/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"

namespace mlir::triton::xla {

static std::optional<gpu::CTALayoutAttr> getCTALayoutOrError(
    AsmParser &parser, std::optional<SmallVector<unsigned>> CTAsPerCGA,
    std::optional<SmallVector<unsigned>> CTASplitNum,
    std::optional<SmallVector<unsigned>> CTAOrder, unsigned rank) {
  if (CTAsPerCGA && CTASplitNum && CTAOrder) {
    return gpu::CTALayoutAttr::get(parser.getContext(), *CTAsPerCGA,
                                   *CTASplitNum, *CTAOrder);
  }
  if (!CTAsPerCGA && !CTASplitNum && !CTAOrder) {
    return gpu::CTALayoutAttr::getDefault(parser.getContext(), rank);
  }
  parser.emitError(parser.getNameLoc(),
                   "CTAsPerCGA, CTASplitNum, and CTAOrder "
                   "must all be present or all be absent");
  return std::nullopt;
}

static LogicalResult parseIntAttrValue(AsmParser &parser, Attribute attr,
                                       unsigned &value, StringRef desc) {
  auto intAttr = mlir::dyn_cast<IntegerAttr>(attr);
  if (!intAttr) {
    parser.emitError(parser.getNameLoc(), "expected an integer type in ")
        << desc;
    return failure();
  }
  if (intAttr.getType().isSignedInteger()) {
    int64_t attrVal = intAttr.getSInt();
    if (attrVal < 0) {
      parser.emitError(parser.getNameLoc(),
                       "expected an unsigned integer value in ")
          << desc;
      return failure();
    }
    value = attrVal;
  } else if (intAttr.getType().isSignlessInteger()) {
    int64_t attrVal = intAttr.getInt();
    if (attrVal < 0) {
      parser.emitError(parser.getNameLoc(),
                       "expected an unsigned integer value in ")
          << desc;
      return failure();
    }
    value = attrVal;
  } else {
    value = intAttr.getUInt();
  }
  return success();
}

// parse an array of integers
static LogicalResult parseIntArrayAttr(AsmParser &parser,
                                       const NamedAttribute &attr,
                                       SmallVector<unsigned> &res,
                                       StringRef desc) {
  auto arrayAttr = mlir::dyn_cast<ArrayAttr>(attr.getValue());
  if (!arrayAttr) {
    parser.emitError(parser.getNameLoc(), "expected an array for ") << desc;
    return failure();
  }
  for (Attribute i : arrayAttr) {
    unsigned value;
    if (parseIntAttrValue(parser, i, value, desc).failed()) return failure();
    res.push_back(value);
  }
  return success();
};

static LogicalResult parseUInt(AsmParser &parser, const NamedAttribute &attr,
                               unsigned &value, StringRef desc) {
  return parseIntAttrValue(parser, attr.getValue(), value, desc);
};

//--- SparseDotMetaEncodingAttr ---
unsigned SparseDotMetaEncodingAttr::getTotalElemsPerThread(
    ArrayRef<int64_t> shape, Type eltTy) const {
  constexpr int kMetadataElementsPerWarp = 16;
  auto mmaLayout = mlir::cast<gpu::NvidiaMmaEncodingAttr>(getParent());
  return product<int64_t>(shape) /
         (mmaLayout.getWarpsPerCTA()[0] * kMetadataElementsPerWarp);
}

SmallVector<unsigned> SparseDotMetaEncodingAttr::getElemsPerThread(
    ArrayRef<int64_t> shape, Type eltTy) const {
  llvm_unreachable("getElemsPerThread is not supported for sparse dot meta");
  return SmallVector<unsigned>();
}

SmallVector<unsigned> SparseDotMetaEncodingAttr::getCTAsPerCGA() const {
  return gpu::getCTAsPerCGA(getParent());
}
SmallVector<unsigned> SparseDotMetaEncodingAttr::getCTAOrder() const {
  return gpu::getCTAOrder(getParent());
}
SmallVector<unsigned> SparseDotMetaEncodingAttr::getCTASplitNum() const {
  return gpu::getCTASplitNum(getParent());
}
SmallVector<unsigned> SparseDotMetaEncodingAttr::getWarpsPerCTA() const {
  return gpu::getWarpsPerCTA(getParent());
}
SmallVector<unsigned> SparseDotMetaEncodingAttr::getWarpOrder() const {
  return {1, 0};
}
SmallVector<unsigned> SparseDotMetaEncodingAttr::getThreadsPerWarp() const {
  return gpu::getThreadsPerWarp(getParent());
}
SmallVector<unsigned> SparseDotMetaEncodingAttr::getThreadOrder() const {
  return {1, 0};
}
SmallVector<unsigned> SparseDotMetaEncodingAttr::getSizePerThread() const {
  return gpu::getSizePerThread(getParent());
}
SmallVector<unsigned> SparseDotMetaEncodingAttr::getShapePerCTATile(
    ArrayRef<int64_t> tensorShape) const {
  return gpu::getShapePerCTATile(getParent(), tensorShape);
}
std::optional<LinearLayout> SparseDotMetaEncodingAttr::toLinearLayout(
    ArrayRef<int64_t> shape) const {
  return gpu::toLinearLayout(shape, getParent());
}

// Attribute SparseDotMetaEncodingAttr::parse(AsmParser &parser, Type type) {
//   if (parser.parseLess().failed())
//     return {};
//   // Parse the data as a dictionary
//   DictionaryAttr dict;
//   if (parser.parseAttribute(dict).failed())
//     return {};
//   if (parser.parseGreater().failed())
//     return {};

//   SmallVector<unsigned> sizePerThread;
//   SmallVector<unsigned> threadsPerWarp;
//   SmallVector<unsigned> warpsPerCTA;
//   SmallVector<unsigned> order;
//   std::optional<SmallVector<unsigned>> CTAsPerCGA;
//   std::optional<SmallVector<unsigned>> CTASplitNum;
//   std::optional<SmallVector<unsigned>> CTAOrder;

//   for (const NamedAttribute &attr : dict) {
//     if (attr.getName() == "sizePerThread") {
//       if (parseIntArrayAttr(parser, attr, sizePerThread,
//                             "number of elements per thread")
//               .failed())
//         return {};
//     } else if (attr.getName() == "threadsPerWarp") {
//       if (parseIntArrayAttr(parser, attr, threadsPerWarp,
//                             "number of threads per warp")
//               .failed())
//         return {};
//     } else if (attr.getName() == "warpsPerCTA") {
//       if (parseIntArrayAttr(parser, attr, warpsPerCTA,
//                             "number of warps per CTA")
//               .failed())
//         return {};
//     } else if (attr.getName() == "order") {
//       if (parseIntArrayAttr(parser, attr, order, "order").failed())
//         return {};
//     } else if (attr.getName() == "CTAsPerCGA") {
//       if (parseIntArrayAttr(parser, attr, CTAsPerCGA.emplace(), "CTAsPerCGA")
//               .failed())
//         return {};
//     } else if (attr.getName() == "CTASplitNum") {
//       if (parseIntArrayAttr(parser, attr, CTASplitNum.emplace(),
//       "CTASplitNum")
//               .failed())
//         return {};
//     } else if (attr.getName() == "CTAOrder") {
//       if (parseIntArrayAttr(parser, attr, CTAOrder.emplace(), "CTAOrder")
//               .failed())
//         return {};
//     } else {
//       parser.emitError(parser.getNameLoc(), "unexpected key: ")
//           << attr.getName().strref();
//       return {};
//     }
//   }

//   std::optional<gpu::CTALayoutAttr> CTALayout = getCTALayoutOrError(
//       parser, CTAsPerCGA, CTASplitNum, CTAOrder,
//       /*rank=*/sizePerThread.size());
//   if (!CTALayout.has_value())
//     return {};

//   return parser.getChecked<SparseDotMetaEncodingAttr>(parser.getContext(),
//                                                 sizePerThread,
//                                                 threadsPerWarp, warpsPerCTA,
//                                                 order, *CTALayout);
// }

}  // namespace mlir::triton::xla
