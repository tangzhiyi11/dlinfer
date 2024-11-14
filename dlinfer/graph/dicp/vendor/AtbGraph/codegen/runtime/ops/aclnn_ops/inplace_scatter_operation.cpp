#include "inplace_scatter_operation.h"

#include "aclnnop/aclnn_scatter.h"
#include "log.h"

namespace dicp {

const int NUM1 = 1;
const int NUM3 = 3;

AclNnInplaceScatterOperation::AclNnInplaceScatterOperation(const std::string& name, int64_t dim, int64_t reduce) : AclNnOperation(name), dim_(dim), reduce_(reduce) {}

AclNnInplaceScatterOperation::~AclNnInplaceScatterOperation() {}

atb::Status AclNnInplaceScatterOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    DICP_LOG(INFO) << opName_ << " infer shape start";
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;

    for (size_t i = 0; i < outTensorDescs.at(0).shape.dimNum; ++i) {
        outTensorDescs.at(0).shape.dims[i] = inTensorDescs.at(0).shape.dims[i];
    }
    DICP_LOG(INFO) << opName_ << " infer shape end";
    return 0;
}

uint32_t AclNnInplaceScatterOperation::GetInputNum() const { return NUM3; }

uint32_t AclNnInplaceScatterOperation::GetOutputNum() const { return NUM1; }

int AclNnInplaceScatterOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    DICP_LOG(INFO) << opName_ << " AclNnInplaceScatterGetWorkspaceSize start";

    int ret = aclnnInplaceScatterGetWorkspaceSize(aclInTensors_.at(0).tensor, dim_, aclInTensors_.at(1).tensor, aclInTensors_.at(2).tensor, reduce_, &workspaceSize, &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " AclNnInplaceScatterGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize << ", aclExecutor:" << aclExecutor_;

    return ret;
}

int AclNnInplaceScatterOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    DICP_LOG(INFO) << opName_ << " AclNnInplaceScatter start";
    int ret = aclnnInplaceScatter(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " AclNnInplaceScatter end, ret:" << ret;
    return ret;
}

}  // namespace dicp
