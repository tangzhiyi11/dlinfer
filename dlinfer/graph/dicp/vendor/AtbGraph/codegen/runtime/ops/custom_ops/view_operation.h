#pragma once

#include <string>

#include "atb/operation.h"

namespace dicp {

class ViewOperation : public atb::Operation {
public:
    explicit ViewOperation(const std::string& name);
    ~ViewOperation() override;
    std::string GetName() const override;
    atb::Status Setup(const atb::VariantPack& variantPack, uint64_t& workspaceSize, atb::Context* context) override;
    atb::Status Execute(const atb::VariantPack& variantPack, uint8_t* workspace, uint64_t workspaceSize, atb::Context* context) override;

protected:
    std::string opName_;
    std::vector<int64_t> viewShape_;
};
}  // namespace dicp
