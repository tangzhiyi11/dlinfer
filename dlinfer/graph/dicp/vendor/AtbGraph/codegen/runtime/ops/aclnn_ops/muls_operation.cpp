#include "muls_operation.h"

#include <cstdint>

#include "aclnnop/aclnn_mul.h"
#include "utils/global_dict.h"
#include "utils/log.h"
#include "utils/misc.h"

namespace dicp {

const int NUM1 = 1;

AclNnMulsOperation::AclNnMulsOperation(const std::string& name, const std::string& value, const std::string& dtype) : AclNnOperation(name), value_(value) {
    if (!value_.empty() && !std::isdigit(value_[0])) {
        need_update_value_ = true;
        other_ = DICPScalar(0.0f, dtype);
    } else {
        need_update_value_ = false;
        other_ = DICPScalar(std::stof(value_), dtype);
    }
    aclOther_ = aclCreateScalar(other_.getValuePtr(), other_.getDataType());
}

AclNnMulsOperation::~AclNnMulsOperation() {
    if (aclOther_ != nullptr) {
        aclDestroyScalar(aclOther_);
    }
}

atb::Status AclNnMulsOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
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

uint32_t AclNnMulsOperation::GetInputNum() const { return NUM1; }

uint32_t AclNnMulsOperation::GetOutputNum() const { return NUM1; }

int AclNnMulsOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    DICP_LOG(INFO) << opName_ << " aclnnMulsGetWorkspaceSize start";

    if (need_update_value_) {
        auto& global_dict = GetGlobalDictData();
        auto it = global_dict.find(value_);
        if (it != global_dict.end()) {
            other_.update_value(std::to_string(it->second));
            if (aclOther_ != nullptr) {
                aclDestroyScalar(aclOther_);
            }
            aclOther_ = aclCreateScalar(other_.getValuePtr(), other_.getDataType());
        } else {
            DICP_LOG(ERROR) << "Cannot find key " << it->second << " in global_dict";
        }
    }

    int ret = aclnnMulsGetWorkspaceSize(aclInTensors_.at(0).tensor, aclOther_, aclOutTensors_.at(0).tensor, &workspaceSize, &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " aclnnMulsGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize << ", aclExecutor:" << aclExecutor_;

    return ret;
}

int AclNnMulsOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    DICP_LOG(INFO) << opName_ << " aclnnMuls start";
    int ret = aclnnMuls(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " aclnnMuls end, ret:" << ret;
    return ret;
}

atb::Operation* AclNnMulsOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    std::string value;
    std::string dtype;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("value")) {
        value = paramJson["value"].get<std::string>();
    }
    if (paramJson.contains("dtype")) {
        dtype = paramJson["dtype"].get<std::string>();
    }
    DICP_LOG(INFO) << "AclNnMulsOperation: name: " << opName << " value:" << value << " dtype:" << dtype;
    atb::Operation* op = new AclNnMulsOperation(opName, value, dtype);
    return op;
}

REGISTER_OPERATION(AclNnMulsOperation, AclNnMulsOperationCreate);

}  // namespace dicp
