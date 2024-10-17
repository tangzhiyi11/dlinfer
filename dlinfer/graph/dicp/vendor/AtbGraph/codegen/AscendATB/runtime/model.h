#pragma once
#include <acl/acl.h>
#include <atb/context.h>
#include <atb/operation.h>
#include <nlohmann/json.hpp>
#include <taskflow/taskflow.hpp>
#include <torch/torch.h>

#include <atomic>
#include <condition_variable>
#include <functional>
#include <map>
#include <mutex>
#include <queue>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "log.h"
#include "nlohmann/json.hpp"
#include "timer.h"

namespace dicp {

template <typename T>
T getValue(const nlohmann::json& node, const std::string& key) {
    try {
        return node.at(key).get<T>();
    } catch (const std::exception& e) {
        DICP_LOG(ERROR) << "Error: " << e.what();
        DICP_LOG(ERROR) << "JSON Node: " << node.dump(4);
        throw std::runtime_error("getValue failed!");
    }
}

class Model {
public:
    using ReshapeFunc = std::function<void(const atb::Dims& oldDims, atb::Dims& newDims)>;
    enum TensorType {
        INTERMEDIATE_TENSOR = 0,
        NOT_INTERMEDIATE_TENSOR,
    };

    struct Node {
        std::shared_ptr<atb::Operation> operation;
        std::vector<atb::Tensor*> inTensors;
        std::vector<atb::Tensor*> outTensors;
        atb::VariantPack variantPack;
        atb::SVector<ReshapeFunc> inTensorReshapeFuncs;
        atb::SVector<TensorType> inTensorTypes;
        atb::SVector<TensorType> outTensorTypes;
        std::unordered_map<int, int> inplaceIndices;
        uint64_t workspaceSize = 0;
        void* workspace = nullptr;
    };

    struct Graph {
        std::vector<atb::Tensor> inTensors;
        std::vector<atb::Tensor> outTensors;
        std::vector<atb::Tensor> internalTensors;
        std::vector<Node> nodes;
        std::map<uint64_t, std::set<atb::Tensor*>> maxNodeIdTensorMap;
        void Init();
        std::string ToString() const;

    private:
        void InitTensorType();
        bool IsInternalTensor(const atb::Tensor* tensor);
        void InitTensorMaxNodeMap();
    };

    Model(const std::string& modelId, const std::string& modelPath);
    virtual ~Model();
    int64_t Init();

    uint32_t GetInputNum() const;
    uint32_t GetOutputNum() const;

    atb::Status Execute(atb::Context* context, std::vector<atb::Tensor>& inTensors,
                        std::vector<atb::Tensor>& outTensors, const std::string& param);
    int64_t BuildGraph();
    atb::Tensor CreateInternalTensorFromDesc(const atb::TensorDesc& tensorDesc);
    void CreateSingleOperation(const nlohmann::json& paramJson, Node& node);
    void CreateGraphOperation(const nlohmann::json& paramJson, Node& node);

    atb::Status ParseParam(const std::string& param);
    atb::Status BindParamHostTensor(uint32_t nodeId);

    bool IsTensorDescEqual(const atb::TensorDesc& tensorDesc, const atb::Tensor& atTensor) const;
    void BuildNodeVariantPack(int nodeId);
    atb::Status ExecuteNode(int nodeId);
    void ClearInternalTensors();
    atb::Tensor MallocInternalTensor(atb::Tensor* outTensor, size_t nodeId, size_t outTensorId,
                                     const atb::TensorDesc& tensorDesc);
    void FreeInternalTensor(void* tensorDeviceData);
    void GetModelTensorNameList(nlohmann::json& modelJson,
                                std::map<atb::Tensor*, std::string>& tensorNameMap);
    void GetNodeTopoInfo(nlohmann::json& nodeJson, const Node& opNode,
                         const std::map<atb::Tensor*, std::string> tensorNameMap);
    std::string GetModelTopoInfo();

private:
    std::string modelId_;
    std::string modelPath_;
    Graph graph_;

    atb::Context* context_;

    int32_t currentDevId_ = 0;
    std::vector<std::pair<atb::Tensor, bool>> internalTensors_;
    std::vector<atb::Tensor*> nodeOutTensors_;

    tf::Executor executor_;  // Taskflow 执行器
    tf::Taskflow taskflow_;  // Taskflow 任务流

    std::vector<torch::Tensor> atInternalTensors_;

    std::unordered_map<std::string, int> tensorsMap_;
    std::unordered_map<std::string, int> inputTensorsMap_;
    std::unordered_map<std::string, int> outputTensorsMap_;
    std::unordered_map<std::string, int> internalTensorsMap_;
    std::unordered_map<int32_t, std::unordered_map<int32_t, std::vector<int32_t>>>
        nodeHostTensorMap_;
};
}  // namespace dicp
