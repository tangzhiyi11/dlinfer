import torch
import os
from contextlib import nullcontext

import torch.distributed

os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
os.environ.setdefault('MASTER_PORT', '28401')
os.environ.setdefault('PROFILING_FILEPATH', os.getcwd() + '/profile')

def init_profiler_with_rank(rank):
    # if True:
    #     return nullcontext()
    import torch_npu
    profiler = torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU
            ],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(os.getenv('PROFILING_FILEPATH')),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_flops=False,
            with_modules=False,
            experimental_config=None)
    return profiler


# @torch.compile(dynamic=False, backend="atbgraph")
# def compile_func(lhs, rhs):
#     for i in range(10):
#         lhs = torch.ops.atb.linear.default(lhs, rhs, None, False, True)
#         lhs = torch.ops.atb.allreduce.default(lhs, "sum")
#     return lhs


@torch.compile(dynamic=False, backend="atbgraph")
def compile_func(lhs, rhs):
    for i in range(2):
        lhs = torch.ops.dlinfer.linear.default(lhs, rhs, None, True)
    return lhs


def real_func(lhs, rhs):
    for i in range(2):
        lhs = torch.nn.functional.linear(lhs, rhs)
        torch.distributed.all_reduce(lhs)
    return lhs

def test_overlap(rank, world_size):
    # target_func = real_func
    target_func = compile_func
    import dlinfer
    import dlinfer.ops
    from dlinfer.graph.dicp.vendor.AtbGraph import ext_ops
    import torch.distributed as dist
    torch.cuda.set_device(rank)
    # profiler = init_profiler_with_rank(rank)
    profiler = nullcontext()
    with profiler:
        dist.init_process_group(backend="nccl",
                                rank=rank,
                                world_size=world_size)
        print(f"rank {rank} init")
        kwargs = dict(device="cuda", dtype=torch.float16)

        # make input
        rhs1 = torch.randn(1023, 1023, **kwargs) / 32.
        lhs1 = torch.randn(1023, 1023, **kwargs) / 32.

        rhs2 = torch.randn(1024, 1024, **kwargs)
        lhs2 = torch.randn(1024, 1024, **kwargs)

        stream1 = torch.cuda.Stream()
        stream2 = torch.cuda.Stream()

        # print(f"rank {rank} before lhs1: {lhs1}")
        for i in range(1):
            # with torch.cuda.stream(stream1):
            res_compile = target_func(lhs1, rhs1)
            # with torch.cuda.stream(stream2):
            #     lhs2 = target_func(lhs2, rhs2)
        torch.cuda.synchronize()
        res_eager = real_func(lhs1, rhs1)
        torch.cuda.synchronize()
        if rank == 0:
            print(f"res_compile: {res_compile}")
            print(f"res_eager: {res_eager}")
        # print(f"rank {rank} after lhs1: {lhs1}")


        print(f"rank {rank} finished")

def main():
    import multiprocessing as mp

    world_size = 2
    sub_p = []
    for rank in range(world_size):
        p = mp.Process(target=test_overlap, args=(rank, world_size))
        p.start()
        sub_p.append(p)

    for p in sub_p:
        p.join()

if __name__ == "__main__":
    main()
