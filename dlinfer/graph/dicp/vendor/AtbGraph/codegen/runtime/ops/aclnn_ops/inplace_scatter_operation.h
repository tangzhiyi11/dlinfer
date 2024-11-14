#pragma once

#include "acl_nn_operation.h"


namespace dicp {

class AclNnInplaceScatterOperation : public AclNnOperation {
public:
    explicit AclNnInplaceScatterOperation(const std::string& name, int64_t dim, int64_t reduce);
    ~AclNnInplaceScatterOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int64_t dim_;
    int64_t reduce_;
    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;
};

inline atb::Operation* AclNnInplaceScatterOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    int64_t dim = 0;
    int64_t reduce = 0;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("dim")) {
        dim = paramJson["dim"].get<int64_t>();
    }
    if (paramJson.contains("reduceType")) {
        reduce = paramJson["reduceType"].get<int64_t>();
    }
    DICP_LOG(INFO) << "AclNnInplaceScatterOperation: name: " << opName << " dim:" << dim << " reduce:" << reduce;
    atb::Operation* op = new AclNnInplaceScatterOperation(opName, dim, reduce);
    return op;
}

}  // namespace dicp
