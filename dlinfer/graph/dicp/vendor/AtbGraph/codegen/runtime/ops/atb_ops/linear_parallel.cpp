#include "atb_ops.h"
namespace dicp {

atb::Operation* LinearParallelOperationCreate(const nlohmann::json& paramJson) {
    atb::infer::LinearParallelParam param;
    if (paramJson.contains("rank")) {
        param.rank = paramJson["rank"].get<int32_t>();
    }
    if (paramJson.contains("rankSize")) {
        param.rankSize = paramJson["rankSize"].get<int32_t>();
    }
    if (paramJson.contains("rankRoot")) {
        param.rankRoot = paramJson["rankRoot"].get<int32_t>();
    }
    if (paramJson.contains("hasResidual")) {
        param.hasResidual = paramJson["hasResidual"].get<bool>();
    }
    if (paramJson.contains("parallelType")) {
        auto type = paramJson["parallelType"].get<int32_t>();
        param.type = static_cast<atb::infer::LinearParallelParam::ParallelType>(type);
    }
    if (paramJson.contains("backend")) {
        param.backend = paramJson["backend"].get<std::string>();
    }
    if (paramJson.contains("commDomain")) {
        param.commDomain = paramJson["commDomain"].get<std::string>();
    }
    if (paramJson.contains("commMode")) {
        auto mode = paramJson["commMode"].get<int32_t>();
        param.commMode = static_cast<atb::infer::CommMode>(mode);
    }
    if (paramJson.contains("rankTableFile")) {
        param.rankTableFile = paramJson["rankTableFile"].get<std::string>();
    }
    DICP_LOG(INFO) << "LinearParallelParam: rank:" << param.rank << ", rankSize:" << param.rankSize << ", outDataType:" << param.outDataType
                   << " backend:" << param.backend << ", commDomain:" << param.commDomain << ", commMode:" << param.commMode << ", rankTableFile"
                   << param.rankTableFile;
    atb::Operation* op = nullptr;

    CREATE_OPERATION_NO_RETURN(param, &op);
    return op;
}

REGISTER_ATB_OPERATION("LinearParallelOperation", LinearParallelOperationCreate);

}  // namespace dicp
