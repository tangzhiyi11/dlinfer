#include "model.h"

#include <acl/acl.h>
#include <atb/types.h>
#include <atb/utils.h>
#include <nlohmann/json.hpp>

#include <fstream>

#include "config.h"
#include "log.h"
#include "operation_creator.h"
#include "tensor_utils.h"
#include "workspace.h"

namespace dicp {
static bool IsTensorDimsEqual(const atb::Dims& left, const atb::Dims& other) {
    if (left.dimNum != other.dimNum) {
        return false;
    }

    for (uint64_t i = 0; i < left.dimNum; ++i) {
        if (left.dims[i] != other.dims[i]) {
            return false;
        }
    }

    return true;
}

std::string Model::Graph::ToString() const {
    std::stringstream ss;
    for (size_t i = 0; i < inTensors.size(); ++i) {
        ss << "inTensors[" << i << "]:" << &inTensors.at(i) << " "
           << tensor_utils::TensorToString(inTensors.at(i)) << std::endl;
    }
    for (size_t i = 0; i < outTensors.size(); ++i) {
        ss << "outTensors[" << i << "]:" << &outTensors.at(i) << " "
           << tensor_utils::TensorToString(outTensors.at(i)) << std::endl;
    }
    for (size_t i = 0; i < internalTensors.size(); ++i) {
        ss << "internalTensors[" << i << "]:" << &internalTensors.at(i) << " "
           << tensor_utils::TensorToString(internalTensors.at(i)) << std::endl;
    }
    ss << "nodes:" << nodes.size() << std::endl;

    for (size_t i = 0; i < nodes.size(); ++i) {
        auto& node = nodes.at(i);
        ss << "node[" << i << "] operation:" << node.operation.get()
           << ", operationName:" << node.operation->GetName() << std::endl;
        for (auto tensorIt : node.inTensors) {
            ss << "node[" << i << "] inTensor:" << tensorIt << " "
               << tensor_utils::TensorToString(*tensorIt) << std::endl;
        }
        for (auto tensorIt : node.outTensors) {
            ss << "node[" << i << "] outTensor:" << tensorIt << " "
               << tensor_utils::TensorToString(*tensorIt) << std::endl;
        }
    }
    return ss.str();
}

void Model::Graph::Init() {
    for (size_t i = 0; i < nodes.size(); i++) {
        auto& node = nodes.at(i);
        node.variantPack.inTensors.resize(node.inTensors.size());
        node.variantPack.outTensors.resize(node.outTensors.size());
    }
    InitTensorType();
    InitTensorMaxNodeMap();
}

void Model::Graph::InitTensorType() {
    for (auto& node : nodes) {
        node.inTensorTypes.resize(node.inTensors.size());
        node.outTensorTypes.resize(node.outTensors.size());
        for (size_t i = 0; i < node.inTensors.size(); ++i) {
            node.inTensorTypes.at(i) = IsInternalTensor(node.inTensors.at(i))
                                           ? Model::INTERMEDIATE_TENSOR
                                           : Model::NOT_INTERMEDIATE_TENSOR;
        }
        for (size_t i = 0; i < node.outTensors.size(); ++i) {
            node.outTensorTypes.at(i) = IsInternalTensor(node.outTensors.at(i))
                                            ? Model::INTERMEDIATE_TENSOR
                                            : Model::NOT_INTERMEDIATE_TENSOR;
        }
    }
}

bool Model::Graph::IsInternalTensor(const atb::Tensor* tensor) {
    for (auto& internalTensor : internalTensors) {
        if (&internalTensor == tensor) {
            return true;
        }
    }

    return false;
}

void Model::Graph::InitTensorMaxNodeMap() {
    std::map<atb::Tensor*, uint64_t> tensorMaxNodeIdMap;
    maxNodeIdTensorMap.clear();

    for (size_t i = 0; i < internalTensors.size(); ++i) {
        atb::Tensor& internalTensor = internalTensors[i];
        uint64_t maxNodeId = 0;
        uint64_t dependNodeCount = 0;
        for (size_t nodeId = 0; nodeId < nodes.size(); ++nodeId) {
            auto& node = nodes.at(nodeId);
            for (auto inTensorIt : node.inTensors) {
                if (&internalTensor == inTensorIt) {
                    maxNodeId = nodeId;
                    dependNodeCount++;
                }
            }
        }
        tensorMaxNodeIdMap[&internalTensor] = maxNodeId;
        DICP_LOG_IF(dependNodeCount == 0, ERROR)
            << "runner graph internal tensor[" << i << "] dependNodeCount is 0, graph wrong";
        maxNodeIdTensorMap[maxNodeId].insert(&internalTensor);
    }
}

uint32_t Model::GetInputNum() const { return graph_.inTensors.size(); }

uint32_t Model::GetOutputNum() const { return graph_.outTensors.size(); }

atb::Tensor Model::CreateInternalTensorFromDesc(const atb::TensorDesc& tensorDesc) {
    torch::Tensor newAtTensor = tensor_utils::CreateAtTensorFromTensorDesc(tensorDesc);
    atInternalTensors_.push_back(newAtTensor);
    return tensor_utils::AtTensor2Tensor(newAtTensor);
}

Model::Model(const std::string& modelId, const std::string& modelPath)
    : modelId_(modelId), modelPath_(modelPath) {
    aclrtGetDevice(&currentDevId_);
}

Model::~Model() {}

int64_t Model::Init() {
    int64_t atbStatus = BuildGraph();
    graph_.Init();
    DICP_LOG(DEBUG) << modelId_ << " init graph:\n" << graph_.ToString();
    return atbStatus;
}

int64_t Model::BuildGraph() {
    // get json
    std::ifstream f(modelPath_);
    auto paramJson = nlohmann::json::parse(f);

    // parse json
    auto graphInputNames = getValue<std::vector<std::string>>(paramJson, "inputNames");
    auto graphOutputNames = getValue<std::vector<std::string>>(paramJson, "outputNames");
    auto graphInternalNames = getValue<std::vector<std::string>>(paramJson, "internalNames");

    int tensorCount = 0;
    graph_.inTensors.resize(graphInputNames.size());
    graph_.outTensors.resize(graphOutputNames.size());
    graph_.internalTensors.resize(graphInternalNames.size());
    for (unsigned int i = 0; i < graphInputNames.size(); ++i) {
        if (tensorsMap_.count(graphInputNames[i]) > 0) {
            DICP_LOG(ERROR) << "duplicate tensor name: " << graphInputNames[i];
            throw std::runtime_error("duplicate tensor name!");
        }
        tensorsMap_[graphInputNames[i]] = tensorCount++;
        inputTensorsMap_[graphInputNames[i]] = i;
    }
    for (unsigned int i = 0; i < graphOutputNames.size(); ++i) {
        if (tensorsMap_.count(graphOutputNames[i]) > 0) {
            DICP_LOG(ERROR) << "duplicate tensor name: " << graphOutputNames[i];
            throw std::runtime_error("duplicate tensor name");
        }
        tensorsMap_[graphOutputNames[i]] = tensorCount++;
        outputTensorsMap_[graphOutputNames[i]] = i;
    }
    for (unsigned int i = 0; i < graphInternalNames.size(); ++i) {
        if (tensorsMap_.count(graphInternalNames[i]) > 0) {
            DICP_LOG(ERROR) << "duplicate tensor name: " << graphInternalNames[i];
            throw std::runtime_error("duplicate tensor name");
        }
        tensorsMap_[graphInternalNames[i]] = tensorCount++;
        internalTensorsMap_[graphInternalNames[i]] = i;
    }

    for (const auto& node : paramJson["nodes"]) {
        auto nodeType = getValue<std::string>(node, "nodeType");
        auto nodeOp = node["value"];
        Node cur_node;

        if (nodeType == "singleOperation") {
            CreateSingleOperation(nodeOp, cur_node);
        } else if (nodeType == "graphOperation") {
            CreateGraphOperation(nodeOp, cur_node);
        } else {
            DICP_LOG(ERROR) << "invalid node type: " << nodeType;
            throw std::runtime_error("invalid node type!");
        }

        graph_.nodes.push_back(cur_node);
    }

    for (const auto& hostTensor : paramJson["hostTensorNames"]) {
        auto nodeId = getValue<int32_t>(hostTensor, "nodeId");
        auto tensorId = getValue<int32_t>(hostTensor, "tensorId");
        nodeHostTensorMap_[nodeId][tensorId] = {};
    }

    DICP_LOG(INFO) << "Model BuildGraph success";
    return atb::NO_ERROR;
}

atb::Status Model::Execute(atb::Context* context, std::vector<atb::Tensor>& inTensors,
                           std::vector<atb::Tensor>& outTensors, const std::string& param) {
    if (graph_.inTensors.size() != inTensors.size() ||
        graph_.outTensors.size() != outTensors.size()) {
        DICP_LOG(ERROR) << modelId_ << " graph.inTensors.size:" << graph_.inTensors.size()
                        << ", inTensors.size:" << inTensors.size()
                        << ", graph.outTensors.size:" << graph_.outTensors.size()
                        << ", outTensors.size:" << outTensors.size();
        return atb::ERROR_INVALID_GRAPH;
    }

    ParseParam(param);

    ClearInternalTensors();
    nodeOutTensors_.clear();

    context_ = context;
    graph_.inTensors = inTensors;
    graph_.outTensors = outTensors;
    DICP_LOG(INFO) << modelId_ << ", graph:\n" << graph_.ToString();

    taskflow_.clear();

    std::vector<tf::Task> tasks(graph_.nodes.size());

    for (size_t nodeId = 0; nodeId < graph_.nodes.size(); ++nodeId) {
        tasks[nodeId] = taskflow_
                            .emplace([this, nodeId]() {
                                BuildNodeVariantPack(nodeId);
                                BindParamHostTensor(nodeId);
                                atb::Status st = ExecuteNode(nodeId);
                                if (st != 0) {
                                    DICP_LOG(ERROR) << modelId_ << " execute node[" << nodeId
                                                    << "] fail, error code: " << st;
                                }
                            })
                            .name("Node" + std::to_string(nodeId));

        if (nodeId > 0) {
            tasks[nodeId - 1].precede(tasks[nodeId]);
        }
    }

    executor_.run(taskflow_).wait();

    DICP_LOG(INFO) << modelId_ << " execute finshed!";

    return atb::NO_ERROR;
}

atb::Status Model::ExecuteNode(int nodeId) {
    auto& node = graph_.nodes.at(nodeId);

    Timer timer;
    atb::Status st = node.operation->Setup(node.variantPack, node.workspaceSize, context_);
    if (st != 0) {
        DICP_LOG(ERROR) << modelId_ << " setup node[" << nodeId << "] fail, not call execute";
        return st;
    }

    DICP_LOG(INFO) << modelId_ << " get node[" << nodeId
                   << "] workspace size:" << node.workspaceSize;

    if (node.workspaceSize > 0) {
        node.workspace = GetWorkspaceBuffer(node.workspaceSize);
    }

    DICP_LOG(INFO) << modelId_ << "execute node[" << nodeId << "] start";

    st = node.operation->Execute(
        node.variantPack, (uint8_t*)(node.workspace), node.workspaceSize, context_);
    if (st != 0) {
        DICP_LOG(ERROR) << "execute node[" << nodeId << "] fail, error code: " << st;
    }
    return st;
}

atb::Status Model::ParseParam(const std::string& param) {
    nlohmann::json paramJson = nlohmann::json::parse(param);
    for (const auto& node : paramJson["hostTensors"]) {
        auto nodeId = getValue<int32_t>(node, "nodeId");
        auto tensorId = getValue<int32_t>(node, "tensorId");
        auto value = getValue<std::vector<int32_t>>(node, "value");
        nodeHostTensorMap_[nodeId][tensorId] = value;
    }
    return atb::NO_ERROR;
}

atb::Status Model::BindParamHostTensor(uint32_t nodeId) {
    if (nodeHostTensorMap_.count(nodeId) == 0) {
        return atb::NO_ERROR;
    }

    auto& node = graph_.nodes.at(nodeId);
    for (auto& i : nodeHostTensorMap_[nodeId]) {
        node.variantPack.inTensors.at(i.first).hostData = i.second.data();
    }
    DICP_LOG(INFO) << "BindParamHostTensor end";
    return atb::NO_ERROR;
}

void Model::BuildNodeVariantPack(int nodeId) {
    auto& node = graph_.nodes.at(nodeId);
    bool needReshape = node.inTensorReshapeFuncs.size() > 0;

    atb::SVector<atb::TensorDesc> inTensorDescs;
    inTensorDescs.reserve(node.variantPack.inTensors.size());
    inTensorDescs.resize(node.variantPack.inTensors.size());
    for (size_t i = 0; i < node.inTensors.size(); ++i) {
        node.variantPack.inTensors.at(i) = *node.inTensors.at(i);
        inTensorDescs.at(i) = node.inTensors.at(i)->desc;
        if (needReshape) {
            node.inTensorReshapeFuncs.at(i)(node.inTensors.at(i)->desc.shape,
                                            inTensorDescs.at(i).shape);
            node.variantPack.inTensors.at(i).desc.shape = inTensorDescs.at(i).shape;
            node.inTensors.at(i)->desc.shape = inTensorDescs.at(i).shape;
        }
        DICP_LOG(INFO) << modelId_ << " nodes[" << nodeId << "] inTensors[" << i
                       << "]:" << tensor_utils::TensorToString(node.variantPack.inTensors.at(i));
    }

    atb::SVector<atb::TensorDesc> outTensorDescs;
    outTensorDescs.reserve(node.operation->GetOutputNum());
    outTensorDescs.resize(node.operation->GetOutputNum());

    atb::Status st = node.operation->InferShape(inTensorDescs, outTensorDescs);
    DICP_LOG_IF(st != 0, FATAL) << modelId_ << " nodes[" << nodeId << "] "
                                << " infer shape fail, error code: " << st;

    bool hasInplaceOutputs = node.inplaceIndices.size() > 0;
    for (size_t i = 0; i < node.outTensors.size(); ++i) {
        if (hasInplaceOutputs && node.inplaceIndices.count(i) > 0) {
            auto inputIdx = node.inplaceIndices[i];
            node.variantPack.outTensors.at(i) = *node.inTensors.at(inputIdx);
            *node.outTensors.at(i) = node.variantPack.outTensors.at(i);
            continue;
        }

        node.variantPack.outTensors.at(i) = *node.outTensors.at(i);
        if (node.outTensorTypes.at(i) == Model::INTERMEDIATE_TENSOR) {
            node.variantPack.outTensors.at(i) =
                MallocInternalTensor(node.outTensors.at(i), nodeId, i, outTensorDescs.at(i));
            *node.outTensors.at(i) = node.variantPack.outTensors.at(i);
        }
    }

    auto it = graph_.maxNodeIdTensorMap.find(nodeId);
    if (it != graph_.maxNodeIdTensorMap.end()) {
        for (auto tensorIt : it->second) {
            FreeInternalTensor(tensorIt->deviceData);
        }
    }
}

void Model::CreateSingleOperation(const nlohmann::json& paramJson, Node& node) {
    auto opType = getValue<std::string>(paramJson, "type");
    auto opName = getValue<std::string>(paramJson, "name");
    auto opInputNames = getValue<std::vector<std::string>>(paramJson, "inputNames");
    auto opOutputNames = getValue<std::vector<std::string>>(paramJson, "outputNames");
    atb::Operation* op = CreateOperation(opType, paramJson["param"]);

    node.operation.reset(op);
    for (const auto& t : opInputNames) {
        if (inputTensorsMap_.count(t) > 0) {
            node.inTensors.push_back(&graph_.inTensors[inputTensorsMap_[t]]);
        } else if (internalTensorsMap_.count(t) > 0) {
            node.inTensors.push_back(&graph_.internalTensors[internalTensorsMap_[t]]);
        } else if (outputTensorsMap_.count(t) > 0) {
            node.inTensors.push_back(&graph_.outTensors[outputTensorsMap_[t]]);
        } else {
            DICP_LOG(ERROR) << "cannot find name in input/internal: " << t;
            throw std::runtime_error("cannot find name in input/internal!");
        }
    }
    for (const auto& t : opOutputNames) {
        if (outputTensorsMap_.count(t) > 0) {
            node.outTensors.push_back(&graph_.outTensors[outputTensorsMap_[t]]);
        } else if (internalTensorsMap_.count(t) > 0) {
            node.outTensors.push_back(&graph_.internalTensors[internalTensorsMap_[t]]);
        } else {
            DICP_LOG(ERROR) << "cannot find name in output/internal: " << t;
            throw std::runtime_error("cannot find name in input/internal!");
        }
    }
    bool hasReshapeInputs = getValue<bool>(paramJson, "hasReshapeInputs");
    if (hasReshapeInputs) {
        node.inTensorReshapeFuncs.resize(node.inTensors.size());
        auto reshapeInputs = paramJson["reshapeInputs"];
        int count = 0;
        for (const auto& reshapeInput : paramJson["reshapeInputs"]) {
            auto reshapeType = getValue<std::string>(reshapeInput, "reshapeType");
            if (reshapeType == "None") {
                node.inTensorReshapeFuncs.at(count) =
                    [=](const atb::Dims& oldShape, atb::Dims& newShape) { newShape = oldShape; };
            } else if (reshapeType == "view") {
                auto dimNum = getValue<int32_t>(reshapeInput, "dimNum");
                auto dims = getValue<std::vector<int32_t>>(reshapeInput, "dims");
                bool needInferDim = false;
                size_t dimNeedInfer = 0;
                for (size_t i = 0; i < dims.size(); ++i) {
                    if (dims[i] == -1) {
                        needInferDim = true;
                        dimNeedInfer = i;
                        break;
                    }
                }
                node.inTensorReshapeFuncs.at(count) =
                    [=]([[maybe_unused]] const atb::Dims& oldShape, atb::Dims& newShape) {
                        newShape.dimNum = dimNum;
                        if (needInferDim) {
                            int64_t totalValue = 1;
                            int64_t otherProd = 1;
                            for (size_t i = 0; i < oldShape.dimNum; ++i) {
                                totalValue *= oldShape.dims[i];
                            }
                            for (size_t i = 0; i < dims.size(); ++i) {
                                if (i != dimNeedInfer) {
                                    otherProd *= dims[i];
                                }
                            }
                            newShape.dims[dimNeedInfer] = totalValue / otherProd;
                        }
                        for (size_t i = 0; i < dims.size(); ++i) {
                            if (dims[i] == -1) continue;
                            newShape.dims[i] = dims[i];
                        }
                    };
            } else if (reshapeType == "unsqueeze") {
                auto dims = getValue<std::vector<int32_t>>(reshapeInput, "dim");
                node.inTensorReshapeFuncs.at(count) =
                    [=]([[maybe_unused]] const atb::Dims& oldShape, atb::Dims& newShape) {
                        std::vector<int64_t> dimValues;
                        dimValues.resize(oldShape.dimNum);
                        for (size_t i = 0; i < oldShape.dimNum; ++i) {
                            dimValues[i] = oldShape.dims[i];
                        }
                        for (size_t i = 0; i < dims.size(); ++i) {
                            auto pos = dimValues.begin() + dims[i];
                            dimValues.insert(pos, 1);
                        }
                        for (size_t i = 0; i < dimValues.size(); ++i) {
                            newShape.dims[i] = dimValues[i];
                        }
                        newShape.dimNum = dimValues.size();
                    };
            } else if (reshapeType == "squeeze") {
                auto dims = getValue<std::vector<int32_t>>(reshapeInput, "dim");
                node.inTensorReshapeFuncs.at(count) =
                    [=]([[maybe_unused]] const atb::Dims& oldShape, atb::Dims& newShape) {
                        std::vector<int64_t> dimValues;
                        dimValues.resize(oldShape.dimNum);
                        for (size_t i = 0; i < oldShape.dimNum; ++i) {
                            dimValues[i] = oldShape.dims[i];
                        }
                        for (size_t i = 0; i < dims.size(); ++i) {
                            auto pos = dimValues.begin() + dims[i];
                            dimValues.erase(pos);
                        }
                        for (size_t i = 0; i < dimValues.size(); ++i) {
                            newShape.dims[i] = dimValues[i];
                        }
                        newShape.dimNum = dimValues.size();
                    };
            }
            count++;
        }
    }
    bool hasInplaceOutputs = getValue<bool>(paramJson, "hasInplaceOutputs");
    if (hasInplaceOutputs) {
        for (const auto& inplaceTensors : paramJson["inplaceOutputs"]) {
            auto outputIdx = getValue<int32_t>(inplaceTensors, "output_index");
            auto inputIdx = getValue<int32_t>(inplaceTensors, "input_index");
            node.inplaceIndices[outputIdx] = inputIdx;
        }
    }
}

void Model::CreateGraphOperation(const nlohmann::json& paramJson, Node& node) {
    atb::GraphParam graph_param;
    int nodeSize = getValue<int32_t>(paramJson, "nodeSize");
    auto inputNames = getValue<std::vector<std::string>>(paramJson, "inputNames");
    auto outputNames = getValue<std::vector<std::string>>(paramJson, "outputNames");
    auto internalNames = getValue<std::vector<std::string>>(paramJson, "internalNames");
    graph_param.inTensorNum = inputNames.size();
    graph_param.outTensorNum = outputNames.size();
    graph_param.internalTensorNum = internalNames.size();
    graph_param.nodes.resize(nodeSize);

    // graph local tensor ids
    std::unordered_map<std::string, int> graph_tensor_ids;
    int tensorCount = 0;
    for (unsigned int i = 0; i < inputNames.size(); ++i) {
        graph_tensor_ids[inputNames[i]] = tensorCount++;
    }
    for (unsigned int i = 0; i < outputNames.size(); ++i) {
        graph_tensor_ids[outputNames[i]] = tensorCount++;
    }
    for (unsigned int i = 0; i < internalNames.size(); ++i) {
        graph_tensor_ids[internalNames[i]] = tensorCount++;
    }

    int cur_node_index = 0;
    for (const auto& node : paramJson["nodes"]) {
        auto nodeType = getValue<std::string>(node, "nodeType");
        auto nodeOp = node["value"];
        if (nodeType == "singleOperation") {
            auto opType = getValue<std::string>(nodeOp, "type");
            auto opName = getValue<std::string>(nodeOp, "name");
            auto opInputNames = getValue<std::vector<std::string>>(nodeOp, "inputNames");
            auto opOutputNames = getValue<std::vector<std::string>>(nodeOp, "outputNames");
            atb::Operation* op = CreateOperation(opType, nodeOp["param"]);
            graph_param.nodes[cur_node_index].operation = op;
            for (const auto& t : opInputNames) {
                graph_param.nodes[cur_node_index].inTensorIds.push_back(graph_tensor_ids[t]);
            }
            for (const auto& t : opOutputNames) {
                graph_param.nodes[cur_node_index].outTensorIds.push_back(graph_tensor_ids[t]);
            }

            bool hasReshapeInputs = getValue<bool>(nodeOp, "hasReshapeInputs");
            auto& cur_node = graph_param.nodes[cur_node_index];
            if (hasReshapeInputs) {
                cur_node.inTensorReshapeFuncs.resize(cur_node.inTensorIds.size());
                auto reshapeInputs = nodeOp["reshapeInputs"];
                int count = 0;
                for (const auto& reshapeInput : nodeOp["reshapeInputs"]) {
                    auto reshapeType = getValue<std::string>(reshapeInput, "reshapeType");
                    if (reshapeType == "None") {
                        cur_node.inTensorReshapeFuncs.at(count) = [=](const atb::Dims& oldShape,
                                                                      atb::Dims& newShape) {
                            newShape = oldShape;
                        };
                    } else if (reshapeType == "view") {
                        auto dimNum = getValue<int32_t>(reshapeInput, "dimNum");
                        auto dims = getValue<std::vector<int32_t>>(reshapeInput, "dims");
                        bool needInferDim = false;
                        size_t dimNeedInfer = 0;
                        for (size_t i = 0; i < dims.size(); ++i) {
                            if (dims[i] == -1) {
                                needInferDim = true;
                                dimNeedInfer = i;
                                break;
                            }
                        }
                        cur_node.inTensorReshapeFuncs.at(count) =
                            [=]([[maybe_unused]] const atb::Dims& oldShape, atb::Dims& newShape) {
                                newShape.dimNum = dimNum;
                                if (needInferDim) {
                                    int64_t totalValue = 1;
                                    int64_t otherProd = 1;
                                    for (size_t i = 0; i < oldShape.dimNum; ++i) {
                                        totalValue *= oldShape.dims[i];
                                    }
                                    for (size_t i = 0; i < dims.size(); ++i) {
                                        if (i != dimNeedInfer) {
                                            otherProd *= dims[i];
                                        }
                                    }
                                    newShape.dims[dimNeedInfer] = totalValue / otherProd;
                                }
                                for (size_t i = 0; i < dims.size(); ++i) {
                                    if (dims[i] == -1) continue;
                                    newShape.dims[i] = dims[i];
                                }
                            };
                    } else if (reshapeType == "unsqueeze") {
                        auto dims = getValue<std::vector<int32_t>>(reshapeInput, "dim");
                        cur_node.inTensorReshapeFuncs.at(count) =
                            [=]([[maybe_unused]] const atb::Dims& oldShape, atb::Dims& newShape) {
                                std::vector<int64_t> dimValues;
                                dimValues.resize(oldShape.dimNum);
                                for (size_t i = 0; i < oldShape.dimNum; ++i) {
                                    dimValues[i] = oldShape.dims[i];
                                }
                                for (size_t i = 0; i < dims.size(); ++i) {
                                    auto pos = dimValues.begin() + dims[i];
                                    dimValues.insert(pos, 1);
                                }
                                for (size_t i = 0; i < dimValues.size(); ++i) {
                                    newShape.dims[i] = dimValues[i];
                                }
                                newShape.dimNum = dimValues.size();
                            };
                    } else if (reshapeType == "squeeze") {
                        auto dims = getValue<std::vector<int32_t>>(reshapeInput, "dim");
                        cur_node.inTensorReshapeFuncs.at(count) =
                            [=]([[maybe_unused]] const atb::Dims& oldShape, atb::Dims& newShape) {
                                std::vector<int64_t> dimValues;
                                dimValues.resize(oldShape.dimNum);
                                for (size_t i = 0; i < oldShape.dimNum; ++i) {
                                    dimValues[i] = oldShape.dims[i];
                                }
                                for (size_t i = 0; i < dims.size(); ++i) {
                                    auto pos = dimValues.begin() + dims[i];
                                    dimValues.erase(pos);
                                }
                                for (size_t i = 0; i < dimValues.size(); ++i) {
                                    newShape.dims[i] = dimValues[i];
                                }
                                newShape.dimNum = dimValues.size();
                            };
                    }
                    count++;
                }
            }
        } else {
            DICP_LOG(ERROR) << "invalid node type in graph opearation, ndoeType: " << nodeType;
            throw std::runtime_error("invalid node type in graph opearation!");
        }
        cur_node_index++;
    }

    auto hasInferShape = getValue<bool>(paramJson, "hasInferShape");
    if (hasInferShape) {
        auto& inferShape = paramJson["inferShape"];
        auto inferType = getValue<std::string>(inferShape, "type");
        if (inferType == "equal") {
            auto outputByInput = getValue<std::vector<int32_t>>(inferShape, "value");
            graph_param.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc>& inTensorDescs,
                                             atb::SVector<atb::TensorDesc>& outTensorDescs) {
                for (size_t i = 0; i < outTensorDescs.size(); ++i) {
                    outTensorDescs.at(i) = inTensorDescs.at(outputByInput[i]);
                }
                return atb::NO_ERROR;
            };
        }
    }

    atb::Operation* op = nullptr;
    auto st = atb::CreateOperation(graph_param, &op);
    if (st != 0) {
        DICP_LOG(ERROR) << "atb CreateOperation graph failed, st: " << st;
        throw std::runtime_error("atb CreateOperation graph failed!");
    }

    // bind Model tensor ids to graph tensor
    node.operation.reset(op);
    for (const auto& t : inputNames) {
        if (inputTensorsMap_.count(t) > 0) {
            node.inTensors.push_back(&graph_.inTensors[inputTensorsMap_[t]]);
        } else {
            node.inTensors.push_back(&graph_.internalTensors[internalTensorsMap_[t]]);
        }
    }
    for (const auto& t : outputNames) {
        if (outputTensorsMap_.count(t) > 0) {
            node.outTensors.push_back(&graph_.outTensors[outputTensorsMap_[t]]);
        } else {
            node.outTensors.push_back(&graph_.internalTensors[internalTensorsMap_[t]]);
        }
    }
    bool hasInplaceOutputs = getValue<bool>(paramJson, "hasInplaceOutputs");
    if (hasInplaceOutputs) {
        for (const auto& inplaceTensors : paramJson["inplaceOutputs"]) {
            auto outputIdx = getValue<int32_t>(inplaceTensors, "output_index");
            auto inputIdx = getValue<int32_t>(inplaceTensors, "input_index");
            node.inplaceIndices[outputIdx] = inputIdx;
        }
    }
}

bool Model::IsTensorDescEqual(const atb::TensorDesc& tensorDesc,
                              const atb::Tensor& atbTensor) const {
    return atbTensor.desc.dtype == tensorDesc.dtype && atbTensor.desc.format == tensorDesc.format &&
           IsTensorDimsEqual(atbTensor.desc.shape, tensorDesc.shape);
}

void Model::ClearInternalTensors() {
    internalTensors_.clear();
    atInternalTensors_.clear();
}

atb::Tensor Model::MallocInternalTensor(atb::Tensor* outTensor, size_t nodeId, size_t outTensorId,
                                        const atb::TensorDesc& tensorDesc) {
    if (GetConfig().ReuseInternalTensor()) {
        std::vector<atb::Tensor*>::iterator iter =
            std::find(nodeOutTensors_.begin(), nodeOutTensors_.end(), outTensor);
        if (iter != nodeOutTensors_.end()) {
            DICP_LOG(INFO) << modelId_ << " nodeId: " << nodeId
                           << ", out tensor id: " << outTensorId << " write inplace";
            return **iter;
        }
        for (auto& it : internalTensors_) {
            if (it.second) {
                continue;
            }

            if (IsTensorDescEqual(tensorDesc, it.first)) {
                it.second = true;
                DICP_LOG(INFO) << modelId_ << " use old internal tensor";
                return it.first;
            }
        }
    }

    DICP_LOG(INFO) << modelId_ << " create internal tensor, node[" << nodeId << "], outTensor["
                   << outTensorId << "]";
    atb::Tensor newTensor = CreateInternalTensorFromDesc(tensorDesc);
    internalTensors_.push_back(std::make_pair(newTensor, true));
    nodeOutTensors_.push_back(outTensor);
    return newTensor;
}

void Model::FreeInternalTensor(void* tensorDeviceData) {
    if (GetConfig().ReuseInternalTensor()) {
        for (auto& it : internalTensors_) {
            if (it.first.deviceData == tensorDeviceData) {
                it.second = false;
                DICP_LOG(INFO) << modelId_ << " free internal tensor";
                break;
            }
        }
    }
}

void Model::GetModelTensorNameList(nlohmann::json& modelJson,
                                   std::map<atb::Tensor*, std::string>& tensorNameMap) {
    std::string tensorName;

    for (size_t i = 0; i < graph_.inTensors.size(); i++) {
        tensorName = modelId_ + "_input_" + std::to_string(i);
        modelJson["inTensors"].emplace_back(tensorName);
        atb::Tensor& inTensor = graph_.inTensors[i];
        tensorNameMap[&inTensor] = tensorName;
    }

    for (size_t i = 0; i < graph_.outTensors.size(); i++) {
        tensorName = modelId_ + "_output_" + std::to_string(i);
        modelJson["outTensors"].emplace_back(tensorName);
        atb::Tensor& outTensor = graph_.outTensors[i];
        tensorNameMap[&outTensor] = tensorName;
    }

    for (size_t i = 0; i < graph_.internalTensors.size(); i++) {
        tensorName = modelId_ + "_internal_" + std::to_string(i);
        modelJson["internalTensors"].emplace_back(tensorName);
        atb::Tensor& internalTensor = graph_.internalTensors[i];
        tensorNameMap[&internalTensor] = tensorName;
    }
}

void Model::GetNodeTopoInfo(nlohmann::json& nodeJson, const Node& opNode,
                            const std::map<atb::Tensor*, std::string> tensorNameMap) {
    nodeJson["opName"] = opNode.operation->GetName();

    for (auto inTensor : opNode.inTensors) {
        auto it = tensorNameMap.find(inTensor);
        if (it != tensorNameMap.end()) {
            nodeJson["inTensors"].emplace_back(it->second);
        }
    }

    for (auto outTensor : opNode.outTensors) {
        auto it = tensorNameMap.find(outTensor);
        if (it != tensorNameMap.end()) {
            nodeJson["outTensors"].emplace_back(it->second);
        }
    }
}

std::string Model::GetModelTopoInfo() {
    nlohmann::json modelJson;
    modelJson["modelName"] = modelId_;

    std::map<atb::Tensor*, std::string> tensorNameMap;
    GetModelTensorNameList(modelJson, tensorNameMap);

    for (size_t nodeId = 0; nodeId < graph_.nodes.size(); nodeId++) {
        const auto& opNode = graph_.nodes.at(nodeId);
        nlohmann::json nodeJson;
        GetNodeTopoInfo(nodeJson, opNode, tensorNameMap);
        modelJson["nodes"].emplace_back(nodeJson);
    }
    return modelJson.dump();
}

}  // namespace dicp
