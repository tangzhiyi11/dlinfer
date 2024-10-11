#include "batch_matmul_operation.h"

#include <aclnn/acl_meta.h>
#include <securec.h>
#include <syscall.h>
#include <unistd.h>

#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_batch_matmul.h"
#include "log.h"
namespace dicp {
const int DIM0 = 0;
const int DIM1 = 1;
const int DIM2 = 2;
const int DIM3 = 3;
const int NUM1 = 1;
const int NUM2 = 2;
const int NUM3 = 3;
const int NUM4 = 4;

AclNnBatchMatMulOperation::AclNnBatchMatMulOperation(const std::string& name,
                                                     int8_t cubeMathType)
    : AclNnOperation(name) {
  this->cubeMathType = cubeMathType;
}

AclNnBatchMatMulOperation::~AclNnBatchMatMulOperation() {}

atb::Status AclNnBatchMatMulOperation::InferShape(
    const atb::SVector<atb::TensorDesc>& inTensorDescs,
    atb::SVector<atb::TensorDesc>& outTensorDescs) const {
  DICP_LOG(INFO) << opName_ << " infer shape start";
  outTensorDescs.at(0).format = inTensorDescs.at(0).format;
  outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
  outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;

  outTensorDescs.at(0).shape.dims[DIM0] = inTensorDescs.at(0).shape.dims[DIM0];
  outTensorDescs.at(0).shape.dims[DIM1] = inTensorDescs.at(0).shape.dims[DIM1];
  outTensorDescs.at(0).shape.dims[DIM2] = inTensorDescs.at(1).shape.dims[DIM2];
  DICP_LOG(INFO) << opName_ << " infer shape end";
  return 0;
}

uint32_t AclNnBatchMatMulOperation::GetInputNum() const { return NUM2; }

uint32_t AclNnBatchMatMulOperation::GetOutputNum() const { return NUM1; }

int AclNnBatchMatMulOperation::CreateAclTensors(
    const atb::VariantPack& variantPack, AclNnTask& task) {
  DICP_LOG(INFO) << opName_ << " CreateAclTensor start";
  task.aclInTensors.resize(variantPack.inTensors.size());
  for (size_t i = 0; i < task.aclInTensors.size(); ++i) {
    task.aclInTensors[i] = CreateTensor(variantPack.inTensors.at(i));
  }

  DICP_LOG(INFO) << opName_ << " Create aclInTensor end";

  task.aclOutTensors.resize(variantPack.outTensors.size());
  for (size_t i = 0; i < task.aclOutTensors.size(); ++i) {
    task.aclOutTensors[i] = CreateTensor(variantPack.outTensors.at(i));
  }

  DICP_LOG(INFO) << opName_ << " Create aclOutTensor end";
  DICP_LOG(INFO) << opName_ << " CreateAclTensor end";
  return 0;
}

int AclNnBatchMatMulOperation::CallAclGetWorkspace(AclNnTask& task,
                                                   uint64_t& workspaceSize) {
  DICP_LOG(INFO) << opName_ << " aclnnCatGetWorkspaceSize start";

  int ret = aclnnBatchMatMulGetWorkspaceSize(
      task.aclInTensors.at(0).tensor, task.aclInTensors.at(1).tensor,
      task.aclOutTensors.at(0).tensor, this->cubeMathType, &workspaceSize,
      &task.aclExecutor);
  DICP_LOG(INFO) << opName_ << " aclnnCatGetWorkspaceSize end, ret:" << ret
                 << ", workspaceSize:" << workspaceSize
                 << ", aclExecutor:" << task.aclExecutor;

  return ret;
}

int AclNnBatchMatMulOperation::CallAclExecute(uint8_t* workspace,
                                              uint64_t workspaceSize,
                                              aclOpExecutor* aclExecutor,
                                              aclrtStream stream) {
  DICP_LOG(INFO) << opName_ << " aclnnCat start";
  int ret = aclnnBatchMatMul(workspace, workspaceSize, aclExecutor, stream);
  DICP_LOG(INFO) << opName_ << " aclnnCat end, ret:" << ret;
  return ret;
}

AclNnTensor AclNnBatchMatMulOperation::CreateTensor(atb::Tensor atbTensor) {
  AclNnTensor aclNnTensor;
  aclNnTensor.needUpdateTensorDataPtr = true;
  aclNnTensor.atbTensor = atbTensor;
  return aclNnTensor;
}
}  // namespace dicp
