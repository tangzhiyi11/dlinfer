#include "atb_ops.h"
#include "atb/types.h"
#include "third_party/acl/inc/acl/acl_base.h"
#include "third_party/acl/inc/acl/acl_rt.h"
#include "utils/common.h"
#include "utils/misc.h"
#include <cstdint>
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
    DICP_LOG(INFO) << "LinearParallelParam: rank:" << param.rank << ", rankSize:" << param.rankSize << ", outDataType:" << param.outDataType
                   << " backend:" << param.backend;
    atb::Operation* op = nullptr;

    CREATE_OPERATION_NO_RETURN(param, &op);
    return op;
}

REGISTER_ATB_OPERATION("LinearParallelOperation", LinearParallelOperationCreate);

class LinearParallelOperationV2 : public atb::Operation {
public:
    explicit LinearParallelOperationV2(const atb::infer::LinearParallelParam &param);
    ~LinearParallelOperationV2();
    std::string GetName() const override { return opName_; };
    uint32_t GetInputNum() const override { return 2; };
    uint32_t GetOutputNum() const override { return 2; };
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs, atb::SVector<atb::TensorDesc> &outTensorDescs) const override;
    atb::Status Setup(const atb::VariantPack& variantPack, uint64_t& workspaceSize, atb::Context* context) override;
    atb::Status Execute(const atb::VariantPack& variantPack, uint8_t* workspace, uint64_t workspaceSize, atb::Context* context) override;
private:
    atb::Operation* _linearOp1;
    atb::Operation* _linearOp2;
    atb::Operation* _allReduceOp1;
    atb::Operation* _allReduceOp2;
    uint64_t _linear_ws_size1;
    uint64_t _linear_ws_size2;
    uint64_t _allreduce_ws_size1;
    uint64_t _allreduce_ws_size2;
    atb::VariantPack _cached_linear_vp1;
    atb::VariantPack _cached_linear_vp2;
    atb::VariantPack _cached_allreduce_vp1;
    atb::VariantPack _cached_allreduce_vp2;
    uint64_t _workspace_offset;
    aclrtStream _extra_stream;
    aclrtEvent _sync_event;
    std::string opName_ = "LinearParallelOperationV2";
};

LinearParallelOperationV2::LinearParallelOperationV2(const atb::infer::LinearParallelParam &param) {
    if (param.type != param.LINEAR_ALL_REDUCE) {
        throw std::runtime_error("unsupport parallel type for LinearParallelOperationV2");
    }
    atb::infer::LinearParam linear_param {
        .transposeB = param.transWeight,
        .hasBias = param.hasResidual,
    };
    atb::infer::AllReduceParam allreduce_param {
        .rank = param.rank,
        .rankSize = param.rankSize,
        .rankRoot = param.rankRoot,
        .backend = param.backend,
    };
    CREATE_OPERATION_NO_RETURN(linear_param, &_linearOp1);
    CREATE_OPERATION_NO_RETURN(linear_param, &_linearOp2);
    CREATE_OPERATION_NO_RETURN(allreduce_param, &_allReduceOp1);
    CREATE_OPERATION_NO_RETURN(allreduce_param, &_allReduceOp2);
}

LinearParallelOperationV2::~LinearParallelOperationV2() {
    if (_sync_event) {
        aclrtDestroyEvent(_sync_event);
    }
    if (_extra_stream) {
        aclrtDestroyStream(_extra_stream);
    }
    DESTROY_OPERATION_NO_RETURN(_allReduceOp2);
    DESTROY_OPERATION_NO_RETURN(_allReduceOp1);
    DESTROY_OPERATION_NO_RETURN(_linearOp2);
    DESTROY_OPERATION_NO_RETURN(_linearOp1);
}

atb::Status LinearParallelOperationV2::Setup(const atb::VariantPack& variantPack, uint64_t& workspaceSize, atb::Context* context) {
    // lhs data storage offset
    auto input_tensor = variantPack.inTensors.at(0);
    auto weight_tensor = variantPack.inTensors.at(1);
    auto out_tensor = variantPack.outTensors.at(0);
    auto tmp_out_tensor = variantPack.outTensors.at(1);

    auto input_shape = variantPack.inTensors.at(0).desc.shape;
    uint64_t data_inner_size = variantPack.inTensors.at(0).dataSize / input_shape.dims[0];
    uint64_t half_data_offset = (input_shape.dims[0] / 2) * data_inner_size;
    atb::TensorDesc input_first_half_disc = input_tensor.desc;
    input_first_half_disc.shape.dims[0] = input_shape.dims[0] / 2;
    atb::TensorDesc input_last_half_disc = input_tensor.desc;
    input_last_half_disc.shape.dims[0] = input_shape.dims[0] - input_first_half_disc.shape.dims[0];

    // first half
    atb::Tensor input1 = {
        .desc = input_first_half_disc,
        .deviceData = input_tensor.deviceData,
        .dataSize = half_data_offset,
    };
    atb::Tensor linear_out1 = {
        .desc = input_first_half_disc,
        .deviceData = tmp_out_tensor.deviceData,
        .dataSize = half_data_offset,
    };
    atb::Tensor out1 = {
        .desc = input_first_half_disc,
        .deviceData = out_tensor.deviceData,
        .dataSize = half_data_offset,
    };
    _cached_linear_vp1 = {
        .inTensors = { input1, weight_tensor },
        .outTensors = { linear_out1 },
    };
    _cached_allreduce_vp1 = {
        .inTensors = { linear_out1 },
        .outTensors = { out1 },
    };
    DICP_CHECK_ATB_RET(_linearOp1->Setup(_cached_linear_vp1, _linear_ws_size1, context));
    DICP_CHECK_ATB_RET(_allReduceOp1->Setup(_cached_allreduce_vp1, _allreduce_ws_size1, context));

    // second half
    atb::Tensor input2 = {
        .desc = input_last_half_disc,
        .deviceData = reinterpret_cast<void *>(reinterpret_cast<char *>(input_tensor.deviceData) + half_data_offset),
        .dataSize = input_tensor.dataSize - half_data_offset,
    };
    atb::Tensor linear_out2 = {
        .desc = input_last_half_disc,
        .deviceData = reinterpret_cast<void *>(reinterpret_cast<char *>(tmp_out_tensor.deviceData) + half_data_offset),
        .dataSize = input_tensor.dataSize - half_data_offset,
    };
    atb::Tensor out2 = {
        .desc = input_last_half_disc,
        .deviceData = reinterpret_cast<void *>(reinterpret_cast<char *>(out_tensor.deviceData) + half_data_offset),
        .dataSize = input_tensor.dataSize - half_data_offset,
    };
    _cached_linear_vp2 = {
        .inTensors = { input2, weight_tensor },
        .outTensors = { linear_out2 },
    };
    _cached_allreduce_vp2 = {
        .inTensors = { linear_out2 },
        .outTensors = { out2 },
    };
    DICP_CHECK_ATB_RET(_linearOp2->Setup(_cached_linear_vp2, _linear_ws_size2, context));
    DICP_CHECK_ATB_RET(_allReduceOp2->Setup(_cached_allreduce_vp2, _allreduce_ws_size2, context));
    _workspace_offset = std::max(_linear_ws_size1, _allreduce_ws_size1);
    workspaceSize = _workspace_offset + std::max(_linear_ws_size2, _allreduce_ws_size2);
    return atb::NO_ERROR;
}

atb::Status LinearParallelOperationV2::InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                                  atb::SVector<atb::TensorDesc> &outTensorDescs) const {
    DICP_LOG(INFO) << opName_ << " infer shape start";
    for (int i = 0; i < GetOutputNum(); i++) {
        outTensorDescs.at(i).format = inTensorDescs.at(0).format;
        outTensorDescs.at(i).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
        outTensorDescs.at(i).dtype = inTensorDescs.at(0).dtype;
        outTensorDescs.at(i).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
        outTensorDescs.at(i).shape.dims[1] = inTensorDescs.at(1).shape.dims[1];
    }
    DICP_LOG(INFO) << opName_ << " infer shape end";
    return atb::NO_ERROR;
}

atb::Status LinearParallelOperationV2::Execute(const atb::VariantPack &variantPack, uint8_t *workspace, uint64_t workspaceSize, atb::Context *context) {

    if (!_extra_stream) {
        DICP_CHECK_ACL_RET(aclrtCreateStream(&_extra_stream));
    }
    if (!_sync_event) {
        DICP_CHECK_ACL_RET(aclrtCreateEventExWithFlag(&_sync_event, ACL_EVENT_SYNC));
    }
    DICP_CHECK_ATB_RET(_linearOp1->Execute(_cached_linear_vp1, workspace, _workspace_offset, context));
    DICP_CHECK_ATB_RET(_allReduceOp1->Execute(_cached_allreduce_vp1, workspace, _workspace_offset, context));
    auto cur_stream = utils::GetCurrentStream();
    // uncomment following line cause hanging
    // DICP_CHECK_ATB_RET(context->SetExecuteStream(_extra_stream));
    DICP_CHECK_ATB_RET(_linearOp2->Execute(_cached_linear_vp2, workspace + _workspace_offset, workspaceSize - _workspace_offset, context));
    DICP_CHECK_ATB_RET(_allReduceOp2->Execute(_cached_allreduce_vp2, workspace + _workspace_offset, workspaceSize - _workspace_offset, context));
    // DICP_CHECK_ACL_RET(aclrtRecordEvent(_sync_event, _extra_stream));
    // DICP_CHECK_ACL_RET(aclrtStreamWaitEvent(cur_stream, _sync_event));
    DICP_CHECK_ATB_RET(context->SetExecuteStream(cur_stream));
    return atb::NO_ERROR;
}

atb::Operation* LinearParallelOperationV2Create(const nlohmann::json& paramJson) {
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
    DICP_LOG(INFO) << "LinearParallelParam: rank:" << param.rank << ", rankSize:" << param.rankSize << ", outDataType:" << param.outDataType
                   << " backend:" << param.backend;
    atb::Operation* op = new LinearParallelOperationV2(param);
    return op;
}

REGISTER_ATB_OPERATION("LinearParallelOperationV2", LinearParallelOperationV2Create);

}  // namespace dicp
