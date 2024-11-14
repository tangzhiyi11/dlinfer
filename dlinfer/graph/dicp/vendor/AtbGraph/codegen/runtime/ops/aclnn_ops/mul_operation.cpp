#include "mul_operation.h"

#include "aclnnop/aclnn_mul.h"
#include "log.h"
#include "utils.h"

namespace dicp {

const int NUM1 = 1;
const int NUM2 = 2;

AclNnMulOperation::AclNnMulOperation(const std::string& name) : AclNnOperation(name) {
}

AclNnMulOperation::~AclNnMulOperation() {
}

atb::Status AclNnMulOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    DICP_LOG(INFO) << opName_ << " infer shape start";

    auto dimNum  = inTensorDescs.at(0).shape.dimNum > inTensorDescs.at(1).shape.dimNum ? inTensorDescs.at(0).shape.dimNum : inTensorDescs.at(1).shape.dimNum;
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).shape.dimNum = dimNum;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    for (size_t i = 0; i < outTensorDescs.at(0).shape.dimNum; ++i) {
        if (i >= inTensorDescs.at(0).shape.dimNum) {
            outTensorDescs.at(0).shape.dims[i] = inTensorDescs.at(1).shape.dims[i];
        } else if (i >= inTensorDescs.at(1).shape.dimNum ) {
            outTensorDescs.at(0).shape.dims[i] = inTensorDescs.at(0).shape.dims[i];
        } else {
            outTensorDescs.at(0).shape.dims[i] = inTensorDescs.at(0).shape.dims[i] > inTensorDescs.at(1).shape.dims[i] ? inTensorDescs.at(0).shape.dims[i] : inTensorDescs.at(1).shape.dims[i];
        }
    }
    DICP_LOG(INFO) << opName_ << " infer shape end";
    return 0;
}

uint32_t AclNnMulOperation::GetInputNum() const { return NUM2; }

uint32_t AclNnMulOperation::GetOutputNum() const { return NUM1; }

int AclNnMulOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    DICP_LOG(INFO) << opName_ << " AclNnMulGetWorkspaceSize start";

    int ret = aclnnMulGetWorkspaceSize(aclInTensors_.at(0).tensor, aclInTensors_.at(1).tensor, aclOutTensors_.at(0).tensor, &workspaceSize, &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " AclNnMulGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize << ", aclExecutor:" << aclExecutor_;

    return ret;
}

int AclNnMulOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    DICP_LOG(INFO) << opName_ << " AclNnMul start";
    int ret = aclnnMul(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " AclNnMul end, ret:" << ret;
    return ret;
}

}  // namespace dicp
