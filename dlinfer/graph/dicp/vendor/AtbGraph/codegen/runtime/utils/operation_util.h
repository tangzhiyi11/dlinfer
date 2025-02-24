#pragma once

#include <atb/atb_infer.h>
namespace dicp {

#define CREATE_OPERATION(param, operation)                              \
    do {                                                                \
        atb::Status atbStatus = atb::CreateOperation(param, operation); \
        if (atbStatus != atb::NO_ERROR) {                               \
            return atbStatus;                                           \
        }                                                               \
    } while (0)

#define CREATE_OPERATION_NO_RETURN(param, operation)                    \
    do {                                                                \
        atb::Status atbStatus = atb::CreateOperation(param, operation); \
        if (atbStatus != atb::NO_ERROR) {                               \
        }                                                               \
    } while (0)

#define DESTROY_OPERATION(operation)                              \
    do {                                                          \
        atb::Status atbStatus = atb::DestroyOperation(operation); \
        if (atbStatus != atb::NO_ERROR) {                         \
            return atbStatus;                                     \
        }                                                         \
    } while (0)

#define DESTROY_OPERATION_NO_RETURN(operation)                    \
    do {                                                          \
        atb::Status atbStatus = atb::DestroyOperation(operation); \
        if (atbStatus != atb::NO_ERROR) {                         \
        }                                                         \
    } while (0)

}  // namespace dicp
