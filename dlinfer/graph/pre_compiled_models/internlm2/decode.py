import os
import torch
import torch_npu
import random
import json
from dlinfer.graph.dicp.dynamo_bridge.compile import AsyncCompileKernel
from dlinfer.graph.dicp.vendor.AtbGraph.compile_job import AtbCompileJob

kernel_cpp_0 = None

def decode_compile_model():
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, 'decode.json')
    with open(file_path, 'r') as f:
        params_json = json.load(f)
    atb_compile_job = AtbCompileJob(json.dumps(params_json))
    async_compile = AsyncCompileKernel()
    global kernel_cpp_0
    kernel_cpp_0 = async_compile.compile_kernel(atb_compile_job)
    async_compile.wait(globals())
    del async_compile

rope_seqlen_default = torch.ones([1], device="npu", dtype=torch.int32)
def decode_call(model, input_ids, position_ids, past_key_values, attn_metadata, inputs_embeds):
    arg193_1 = model.get_parameter("tok_embeddings.weight")
    arg1_1 = model.get_parameter("layers.0.attention.wqkv.weight")
    arg2_1 = model.get_parameter("layers.0.attention.wo.weight")
    arg4_1 = model.get_parameter("layers.0.feed_forward.gate_up_proj.weight")
    arg5_1 = model.get_parameter("layers.0.feed_forward.w2.weight")
    arg0_1 = model.get_parameter("layers.0.attention_norm.weight")
    arg3_1 = model.get_parameter("layers.0.ffn_norm.weight")
    arg7_1 = model.get_parameter("layers.1.attention.wqkv.weight")
    arg8_1 = model.get_parameter("layers.1.attention.wo.weight")
    arg10_1 = model.get_parameter("layers.1.feed_forward.gate_up_proj.weight")
    arg11_1 = model.get_parameter("layers.1.feed_forward.w2.weight")
    arg6_1 = model.get_parameter("layers.1.attention_norm.weight")
    arg9_1 = model.get_parameter("layers.1.ffn_norm.weight")
    arg13_1 = model.get_parameter("layers.2.attention.wqkv.weight")
    arg14_1 = model.get_parameter("layers.2.attention.wo.weight")
    arg16_1 = model.get_parameter("layers.2.feed_forward.gate_up_proj.weight")
    arg17_1 = model.get_parameter("layers.2.feed_forward.w2.weight")
    arg12_1 = model.get_parameter("layers.2.attention_norm.weight")
    arg15_1 = model.get_parameter("layers.2.ffn_norm.weight")
    arg19_1 = model.get_parameter("layers.3.attention.wqkv.weight")
    arg20_1 = model.get_parameter("layers.3.attention.wo.weight")
    arg22_1 = model.get_parameter("layers.3.feed_forward.gate_up_proj.weight")
    arg23_1 = model.get_parameter("layers.3.feed_forward.w2.weight")
    arg18_1 = model.get_parameter("layers.3.attention_norm.weight")
    arg21_1 = model.get_parameter("layers.3.ffn_norm.weight")
    arg25_1 = model.get_parameter("layers.4.attention.wqkv.weight")
    arg26_1 = model.get_parameter("layers.4.attention.wo.weight")
    arg28_1 = model.get_parameter("layers.4.feed_forward.gate_up_proj.weight")
    arg29_1 = model.get_parameter("layers.4.feed_forward.w2.weight")
    arg24_1 = model.get_parameter("layers.4.attention_norm.weight")
    arg27_1 = model.get_parameter("layers.4.ffn_norm.weight")
    arg31_1 = model.get_parameter("layers.5.attention.wqkv.weight")
    arg32_1 = model.get_parameter("layers.5.attention.wo.weight")
    arg34_1 = model.get_parameter("layers.5.feed_forward.gate_up_proj.weight")
    arg35_1 = model.get_parameter("layers.5.feed_forward.w2.weight")
    arg30_1 = model.get_parameter("layers.5.attention_norm.weight")
    arg33_1 = model.get_parameter("layers.5.ffn_norm.weight")
    arg37_1 = model.get_parameter("layers.6.attention.wqkv.weight")
    arg38_1 = model.get_parameter("layers.6.attention.wo.weight")
    arg40_1 = model.get_parameter("layers.6.feed_forward.gate_up_proj.weight")
    arg41_1 = model.get_parameter("layers.6.feed_forward.w2.weight")
    arg36_1 = model.get_parameter("layers.6.attention_norm.weight")
    arg39_1 = model.get_parameter("layers.6.ffn_norm.weight")
    arg43_1 = model.get_parameter("layers.7.attention.wqkv.weight")
    arg44_1 = model.get_parameter("layers.7.attention.wo.weight")
    arg46_1 = model.get_parameter("layers.7.feed_forward.gate_up_proj.weight")
    arg47_1 = model.get_parameter("layers.7.feed_forward.w2.weight")
    arg42_1 = model.get_parameter("layers.7.attention_norm.weight")
    arg45_1 = model.get_parameter("layers.7.ffn_norm.weight")
    arg49_1 = model.get_parameter("layers.8.attention.wqkv.weight")
    arg50_1 = model.get_parameter("layers.8.attention.wo.weight")
    arg52_1 = model.get_parameter("layers.8.feed_forward.gate_up_proj.weight")
    arg53_1 = model.get_parameter("layers.8.feed_forward.w2.weight")
    arg48_1 = model.get_parameter("layers.8.attention_norm.weight")
    arg51_1 = model.get_parameter("layers.8.ffn_norm.weight")
    arg55_1 = model.get_parameter("layers.9.attention.wqkv.weight")
    arg56_1 = model.get_parameter("layers.9.attention.wo.weight")
    arg58_1 = model.get_parameter("layers.9.feed_forward.gate_up_proj.weight")
    arg59_1 = model.get_parameter("layers.9.feed_forward.w2.weight")
    arg54_1 = model.get_parameter("layers.9.attention_norm.weight")
    arg57_1 = model.get_parameter("layers.9.ffn_norm.weight")
    arg61_1 = model.get_parameter("layers.10.attention.wqkv.weight")
    arg62_1 = model.get_parameter("layers.10.attention.wo.weight")
    arg64_1 = model.get_parameter("layers.10.feed_forward.gate_up_proj.weight")
    arg65_1 = model.get_parameter("layers.10.feed_forward.w2.weight")
    arg60_1 = model.get_parameter("layers.10.attention_norm.weight")
    arg63_1 = model.get_parameter("layers.10.ffn_norm.weight")
    arg67_1 = model.get_parameter("layers.11.attention.wqkv.weight")
    arg68_1 = model.get_parameter("layers.11.attention.wo.weight")
    arg70_1 = model.get_parameter("layers.11.feed_forward.gate_up_proj.weight")
    arg71_1 = model.get_parameter("layers.11.feed_forward.w2.weight")
    arg66_1 = model.get_parameter("layers.11.attention_norm.weight")
    arg69_1 = model.get_parameter("layers.11.ffn_norm.weight")
    arg73_1 = model.get_parameter("layers.12.attention.wqkv.weight")
    arg74_1 = model.get_parameter("layers.12.attention.wo.weight")
    arg76_1 = model.get_parameter("layers.12.feed_forward.gate_up_proj.weight")
    arg77_1 = model.get_parameter("layers.12.feed_forward.w2.weight")
    arg72_1 = model.get_parameter("layers.12.attention_norm.weight")
    arg75_1 = model.get_parameter("layers.12.ffn_norm.weight")
    arg79_1 = model.get_parameter("layers.13.attention.wqkv.weight")
    arg80_1 = model.get_parameter("layers.13.attention.wo.weight")
    arg82_1 = model.get_parameter("layers.13.feed_forward.gate_up_proj.weight")
    arg83_1 = model.get_parameter("layers.13.feed_forward.w2.weight")
    arg78_1 = model.get_parameter("layers.13.attention_norm.weight")
    arg81_1 = model.get_parameter("layers.13.ffn_norm.weight")
    arg85_1 = model.get_parameter("layers.14.attention.wqkv.weight")
    arg86_1 = model.get_parameter("layers.14.attention.wo.weight")
    arg88_1 = model.get_parameter("layers.14.feed_forward.gate_up_proj.weight")
    arg89_1 = model.get_parameter("layers.14.feed_forward.w2.weight")
    arg84_1 = model.get_parameter("layers.14.attention_norm.weight")
    arg87_1 = model.get_parameter("layers.14.ffn_norm.weight")
    arg91_1 = model.get_parameter("layers.15.attention.wqkv.weight")
    arg92_1 = model.get_parameter("layers.15.attention.wo.weight")
    arg94_1 = model.get_parameter("layers.15.feed_forward.gate_up_proj.weight")
    arg95_1 = model.get_parameter("layers.15.feed_forward.w2.weight")
    arg90_1 = model.get_parameter("layers.15.attention_norm.weight")
    arg93_1 = model.get_parameter("layers.15.ffn_norm.weight")
    arg97_1 = model.get_parameter("layers.16.attention.wqkv.weight")
    arg98_1 = model.get_parameter("layers.16.attention.wo.weight")
    arg100_1 = model.get_parameter("layers.16.feed_forward.gate_up_proj.weight")
    arg101_1 = model.get_parameter("layers.16.feed_forward.w2.weight")
    arg96_1 = model.get_parameter("layers.16.attention_norm.weight")
    arg99_1 = model.get_parameter("layers.16.ffn_norm.weight")
    arg103_1 = model.get_parameter("layers.17.attention.wqkv.weight")
    arg104_1 = model.get_parameter("layers.17.attention.wo.weight")
    arg106_1 = model.get_parameter("layers.17.feed_forward.gate_up_proj.weight")
    arg107_1 = model.get_parameter("layers.17.feed_forward.w2.weight")
    arg102_1 = model.get_parameter("layers.17.attention_norm.weight")
    arg105_1 = model.get_parameter("layers.17.ffn_norm.weight")
    arg109_1 = model.get_parameter("layers.18.attention.wqkv.weight")
    arg110_1 = model.get_parameter("layers.18.attention.wo.weight")
    arg112_1 = model.get_parameter("layers.18.feed_forward.gate_up_proj.weight")
    arg113_1 = model.get_parameter("layers.18.feed_forward.w2.weight")
    arg108_1 = model.get_parameter("layers.18.attention_norm.weight")
    arg111_1 = model.get_parameter("layers.18.ffn_norm.weight")
    arg115_1 = model.get_parameter("layers.19.attention.wqkv.weight")
    arg116_1 = model.get_parameter("layers.19.attention.wo.weight")
    arg118_1 = model.get_parameter("layers.19.feed_forward.gate_up_proj.weight")
    arg119_1 = model.get_parameter("layers.19.feed_forward.w2.weight")
    arg114_1 = model.get_parameter("layers.19.attention_norm.weight")
    arg117_1 = model.get_parameter("layers.19.ffn_norm.weight")
    arg121_1 = model.get_parameter("layers.20.attention.wqkv.weight")
    arg122_1 = model.get_parameter("layers.20.attention.wo.weight")
    arg124_1 = model.get_parameter("layers.20.feed_forward.gate_up_proj.weight")
    arg125_1 = model.get_parameter("layers.20.feed_forward.w2.weight")
    arg120_1 = model.get_parameter("layers.20.attention_norm.weight")
    arg123_1 = model.get_parameter("layers.20.ffn_norm.weight")
    arg127_1 = model.get_parameter("layers.21.attention.wqkv.weight")
    arg128_1 = model.get_parameter("layers.21.attention.wo.weight")
    arg130_1 = model.get_parameter("layers.21.feed_forward.gate_up_proj.weight")
    arg131_1 = model.get_parameter("layers.21.feed_forward.w2.weight")
    arg126_1 = model.get_parameter("layers.21.attention_norm.weight")
    arg129_1 = model.get_parameter("layers.21.ffn_norm.weight")
    arg133_1 = model.get_parameter("layers.22.attention.wqkv.weight")
    arg134_1 = model.get_parameter("layers.22.attention.wo.weight")
    arg136_1 = model.get_parameter("layers.22.feed_forward.gate_up_proj.weight")
    arg137_1 = model.get_parameter("layers.22.feed_forward.w2.weight")
    arg132_1 = model.get_parameter("layers.22.attention_norm.weight")
    arg135_1 = model.get_parameter("layers.22.ffn_norm.weight")
    arg139_1 = model.get_parameter("layers.23.attention.wqkv.weight")
    arg140_1 = model.get_parameter("layers.23.attention.wo.weight")
    arg142_1 = model.get_parameter("layers.23.feed_forward.gate_up_proj.weight")
    arg143_1 = model.get_parameter("layers.23.feed_forward.w2.weight")
    arg138_1 = model.get_parameter("layers.23.attention_norm.weight")
    arg141_1 = model.get_parameter("layers.23.ffn_norm.weight")
    arg145_1 = model.get_parameter("layers.24.attention.wqkv.weight")
    arg146_1 = model.get_parameter("layers.24.attention.wo.weight")
    arg148_1 = model.get_parameter("layers.24.feed_forward.gate_up_proj.weight")
    arg149_1 = model.get_parameter("layers.24.feed_forward.w2.weight")
    arg144_1 = model.get_parameter("layers.24.attention_norm.weight")
    arg147_1 = model.get_parameter("layers.24.ffn_norm.weight")
    arg151_1 = model.get_parameter("layers.25.attention.wqkv.weight")
    arg152_1 = model.get_parameter("layers.25.attention.wo.weight")
    arg154_1 = model.get_parameter("layers.25.feed_forward.gate_up_proj.weight")
    arg155_1 = model.get_parameter("layers.25.feed_forward.w2.weight")
    arg150_1 = model.get_parameter("layers.25.attention_norm.weight")
    arg153_1 = model.get_parameter("layers.25.ffn_norm.weight")
    arg157_1 = model.get_parameter("layers.26.attention.wqkv.weight")
    arg158_1 = model.get_parameter("layers.26.attention.wo.weight")
    arg160_1 = model.get_parameter("layers.26.feed_forward.gate_up_proj.weight")
    arg161_1 = model.get_parameter("layers.26.feed_forward.w2.weight")
    arg156_1 = model.get_parameter("layers.26.attention_norm.weight")
    arg159_1 = model.get_parameter("layers.26.ffn_norm.weight")
    arg163_1 = model.get_parameter("layers.27.attention.wqkv.weight")
    arg164_1 = model.get_parameter("layers.27.attention.wo.weight")
    arg166_1 = model.get_parameter("layers.27.feed_forward.gate_up_proj.weight")
    arg167_1 = model.get_parameter("layers.27.feed_forward.w2.weight")
    arg162_1 = model.get_parameter("layers.27.attention_norm.weight")
    arg165_1 = model.get_parameter("layers.27.ffn_norm.weight")
    arg169_1 = model.get_parameter("layers.28.attention.wqkv.weight")
    arg170_1 = model.get_parameter("layers.28.attention.wo.weight")
    arg172_1 = model.get_parameter("layers.28.feed_forward.gate_up_proj.weight")
    arg173_1 = model.get_parameter("layers.28.feed_forward.w2.weight")
    arg168_1 = model.get_parameter("layers.28.attention_norm.weight")
    arg171_1 = model.get_parameter("layers.28.ffn_norm.weight")
    arg175_1 = model.get_parameter("layers.29.attention.wqkv.weight")
    arg176_1 = model.get_parameter("layers.29.attention.wo.weight")
    arg178_1 = model.get_parameter("layers.29.feed_forward.gate_up_proj.weight")
    arg179_1 = model.get_parameter("layers.29.feed_forward.w2.weight")
    arg174_1 = model.get_parameter("layers.29.attention_norm.weight")
    arg177_1 = model.get_parameter("layers.29.ffn_norm.weight")
    arg181_1 = model.get_parameter("layers.30.attention.wqkv.weight")
    arg182_1 = model.get_parameter("layers.30.attention.wo.weight")
    arg184_1 = model.get_parameter("layers.30.feed_forward.gate_up_proj.weight")
    arg185_1 = model.get_parameter("layers.30.feed_forward.w2.weight")
    arg180_1 = model.get_parameter("layers.30.attention_norm.weight")
    arg183_1 = model.get_parameter("layers.30.ffn_norm.weight")
    arg187_1 = model.get_parameter("layers.31.attention.wqkv.weight")
    arg188_1 = model.get_parameter("layers.31.attention.wo.weight")
    arg190_1 = model.get_parameter("layers.31.feed_forward.gate_up_proj.weight")
    arg191_1 = model.get_parameter("layers.31.feed_forward.w2.weight")
    arg186_1 = model.get_parameter("layers.31.attention_norm.weight")
    arg189_1 = model.get_parameter("layers.31.ffn_norm.weight")
    arg192_1 = model.get_parameter("norm.weight")
    arg205_1 = past_key_values[0][0]
    arg206_1 = past_key_values[0][1]
    arg207_1 = past_key_values[1][0]
    arg208_1 = past_key_values[1][1]
    arg209_1 = past_key_values[2][0]
    arg210_1 = past_key_values[2][1]
    arg211_1 = past_key_values[3][0]
    arg212_1 = past_key_values[3][1]
    arg213_1 = past_key_values[4][0]
    arg214_1 = past_key_values[4][1]
    arg215_1 = past_key_values[5][0]
    arg216_1 = past_key_values[5][1]
    arg217_1 = past_key_values[6][0]
    arg218_1 = past_key_values[6][1]
    arg219_1 = past_key_values[7][0]
    arg220_1 = past_key_values[7][1]
    arg221_1 = past_key_values[8][0]
    arg222_1 = past_key_values[8][1]
    arg223_1 = past_key_values[9][0]
    arg224_1 = past_key_values[9][1]
    arg225_1 = past_key_values[10][0]
    arg226_1 = past_key_values[10][1]
    arg227_1 = past_key_values[11][0]
    arg228_1 = past_key_values[11][1]
    arg229_1 = past_key_values[12][0]
    arg230_1 = past_key_values[12][1]
    arg231_1 = past_key_values[13][0]
    arg232_1 = past_key_values[13][1]
    arg233_1 = past_key_values[14][0]
    arg234_1 = past_key_values[14][1]
    arg235_1 = past_key_values[15][0]
    arg236_1 = past_key_values[15][1]
    arg237_1 = past_key_values[16][0]
    arg238_1 = past_key_values[16][1]
    arg239_1 = past_key_values[17][0]
    arg240_1 = past_key_values[17][1]
    arg241_1 = past_key_values[18][0]
    arg242_1 = past_key_values[18][1]
    arg243_1 = past_key_values[19][0]
    arg244_1 = past_key_values[19][1]
    arg245_1 = past_key_values[20][0]
    arg246_1 = past_key_values[20][1]
    arg247_1 = past_key_values[21][0]
    arg248_1 = past_key_values[21][1]
    arg249_1 = past_key_values[22][0]
    arg250_1 = past_key_values[22][1]
    arg251_1 = past_key_values[23][0]
    arg252_1 = past_key_values[23][1]
    arg253_1 = past_key_values[24][0]
    arg254_1 = past_key_values[24][1]
    arg255_1 = past_key_values[25][0]
    arg256_1 = past_key_values[25][1]
    arg257_1 = past_key_values[26][0]
    arg258_1 = past_key_values[26][1]
    arg259_1 = past_key_values[27][0]
    arg260_1 = past_key_values[27][1]
    arg261_1 = past_key_values[28][0]
    arg262_1 = past_key_values[28][1]
    arg263_1 = past_key_values[29][0]
    arg264_1 = past_key_values[29][1]
    arg265_1 = past_key_values[30][0]
    arg266_1 = past_key_values[30][1]
    arg267_1 = past_key_values[31][0]
    arg268_1 = past_key_values[31][1]
    arg196_1 = input_ids
    arg197_1 = position_ids
    arg200_1 = attn_metadata.block_offsets
    arg201_1 = attn_metadata.kv_seqlens
    arg202_1 = attn_metadata.kv_start_indices
    arg195_1 = arg196_1.shape[1]
    arg198_1 = arg200_1.shape[0]
    arg199_1 = arg200_1.shape[1]
    arg203_1 = attn_metadata.block_size
    arg204_1 = attn_metadata.max_kv_seq_len
    arg194_1 = model.rotary_emb.inv_freq

    symInputs = []
    s0 = arg196_1.shape[1]
    s1 = arg200_1.shape[0]
    s2 = arg200_1.shape[1]
    s0 = arg195_1
    symInputs.append('{ "name": "arg195_1", "value": ' + str(s0) + ' }')
    symInputs.append('{ "name": "s0", "value": ' + str(s0) + ' }')
    s1 = arg198_1
    symInputs.append('{ "name": "arg198_1", "value": ' + str(s1) + ' }')
    symInputs.append('{ "name": "s1", "value": ' + str(s1) + ' }')
    s2 = arg199_1
    symInputs.append('{ "name": "arg199_1", "value": ' + str(s2) + ' }')
    symInputs.append('{ "name": "s2", "value": ' + str(s2) + ' }')
    s3 = arg203_1
    symInputs.append('{ "name": "arg203_1", "value": ' + str(s3) + ' }')
    symInputs.append('{ "name": "s3", "value": ' + str(s3) + ' }')
    s4 = arg204_1
    symInputs.append('{ "name": "arg204_1", "value": ' + str(s4) + ' }')
    symInputs.append('{ "name": "s4", "value": ' + str(s4) + ' }')
    rms_norm_64 = torch.empty([1,s0,4096], dtype=torch.bfloat16, device='npu')
    hostTensors = []
    host_tensor_dict = {}
    host_tensor_dict["arg201_1"] = arg201_1.cpu().tolist()
    host_tensor_str_arg201_1 = str(host_tensor_dict["arg201_1"])
    hostTensors.append('{"nodeId": 8, "tensorId": 10, "value": ' + str(host_tensor_str_arg201_1) + ' }')
    hostTensors.append('{"nodeId": 10, "tensorId": 10, "value": ' + str(host_tensor_str_arg201_1) + ' }')
    hostTensors.append('{"nodeId": 12, "tensorId": 10, "value": ' + str(host_tensor_str_arg201_1) + ' }')
    hostTensors.append('{"nodeId": 14, "tensorId": 10, "value": ' + str(host_tensor_str_arg201_1) + ' }')
    hostTensors.append('{"nodeId": 16, "tensorId": 10, "value": ' + str(host_tensor_str_arg201_1) + ' }')
    hostTensors.append('{"nodeId": 18, "tensorId": 10, "value": ' + str(host_tensor_str_arg201_1) + ' }')
    hostTensors.append('{"nodeId": 20, "tensorId": 10, "value": ' + str(host_tensor_str_arg201_1) + ' }')
    hostTensors.append('{"nodeId": 22, "tensorId": 10, "value": ' + str(host_tensor_str_arg201_1) + ' }')
    hostTensors.append('{"nodeId": 24, "tensorId": 10, "value": ' + str(host_tensor_str_arg201_1) + ' }')
    hostTensors.append('{"nodeId": 26, "tensorId": 10, "value": ' + str(host_tensor_str_arg201_1) + ' }')
    hostTensors.append('{"nodeId": 28, "tensorId": 10, "value": ' + str(host_tensor_str_arg201_1) + ' }')
    hostTensors.append('{"nodeId": 30, "tensorId": 10, "value": ' + str(host_tensor_str_arg201_1) + ' }')
    hostTensors.append('{"nodeId": 32, "tensorId": 10, "value": ' + str(host_tensor_str_arg201_1) + ' }')
    hostTensors.append('{"nodeId": 34, "tensorId": 10, "value": ' + str(host_tensor_str_arg201_1) + ' }')
    hostTensors.append('{"nodeId": 36, "tensorId": 10, "value": ' + str(host_tensor_str_arg201_1) + ' }')
    hostTensors.append('{"nodeId": 38, "tensorId": 10, "value": ' + str(host_tensor_str_arg201_1) + ' }')
    hostTensors.append('{"nodeId": 40, "tensorId": 10, "value": ' + str(host_tensor_str_arg201_1) + ' }')
    hostTensors.append('{"nodeId": 42, "tensorId": 10, "value": ' + str(host_tensor_str_arg201_1) + ' }')
    hostTensors.append('{"nodeId": 44, "tensorId": 10, "value": ' + str(host_tensor_str_arg201_1) + ' }')
    hostTensors.append('{"nodeId": 46, "tensorId": 10, "value": ' + str(host_tensor_str_arg201_1) + ' }')
    hostTensors.append('{"nodeId": 48, "tensorId": 10, "value": ' + str(host_tensor_str_arg201_1) + ' }')
    hostTensors.append('{"nodeId": 50, "tensorId": 10, "value": ' + str(host_tensor_str_arg201_1) + ' }')
    hostTensors.append('{"nodeId": 52, "tensorId": 10, "value": ' + str(host_tensor_str_arg201_1) + ' }')
    hostTensors.append('{"nodeId": 54, "tensorId": 10, "value": ' + str(host_tensor_str_arg201_1) + ' }')
    hostTensors.append('{"nodeId": 56, "tensorId": 10, "value": ' + str(host_tensor_str_arg201_1) + ' }')
    hostTensors.append('{"nodeId": 58, "tensorId": 10, "value": ' + str(host_tensor_str_arg201_1) + ' }')
    hostTensors.append('{"nodeId": 60, "tensorId": 10, "value": ' + str(host_tensor_str_arg201_1) + ' }')
    hostTensors.append('{"nodeId": 62, "tensorId": 10, "value": ' + str(host_tensor_str_arg201_1) + ' }')
    hostTensors.append('{"nodeId": 64, "tensorId": 10, "value": ' + str(host_tensor_str_arg201_1) + ' }')
    hostTensors.append('{"nodeId": 66, "tensorId": 10, "value": ' + str(host_tensor_str_arg201_1) + ' }')
    hostTensors.append('{"nodeId": 68, "tensorId": 10, "value": ' + str(host_tensor_str_arg201_1) + ' }')
    hostTensors.append('{"nodeId": 70, "tensorId": 10, "value": ' + str(host_tensor_str_arg201_1) + ' }')
    param = f'{{ "symInputs": [{",".join(symInputs)}], "hostTensors": [{",".join(hostTensors)}] }}'

    inputs = [arg0_1,arg1_1,arg2_1,arg3_1,arg4_1,arg5_1,arg6_1,arg7_1,arg8_1,arg9_1,arg10_1,arg11_1,arg12_1,arg13_1,arg14_1,arg15_1,arg16_1,arg17_1,arg18_1,arg19_1,arg20_1,arg21_1,arg22_1,arg23_1,arg24_1,arg25_1,arg26_1,arg27_1,arg28_1,arg29_1,arg30_1,arg31_1,arg32_1,arg33_1,arg34_1,arg35_1,arg36_1,arg37_1,arg38_1,arg39_1,arg40_1,arg41_1,arg42_1,arg43_1,arg44_1,arg45_1,arg46_1,arg47_1,arg48_1,arg49_1,arg50_1,arg51_1,arg52_1,arg53_1,arg54_1,arg55_1,arg56_1,arg57_1,arg58_1,arg59_1,arg60_1,arg61_1,arg62_1,arg63_1,arg64_1,arg65_1,arg66_1,arg67_1,arg68_1,arg69_1,arg70_1,arg71_1,arg72_1,arg73_1,arg74_1,arg75_1,arg76_1,arg77_1,arg78_1,arg79_1,arg80_1,arg81_1,arg82_1,arg83_1,arg84_1,arg85_1,arg86_1,arg87_1,arg88_1,arg89_1,arg90_1,arg91_1,arg92_1,arg93_1,arg94_1,arg95_1,arg96_1,arg97_1,arg98_1,arg99_1,arg100_1,arg101_1,arg102_1,arg103_1,arg104_1,arg105_1,arg106_1,arg107_1,arg108_1,arg109_1,arg110_1,arg111_1,arg112_1,arg113_1,arg114_1,arg115_1,arg116_1,arg117_1,arg118_1,arg119_1,arg120_1,arg121_1,arg122_1,arg123_1,arg124_1,arg125_1,arg126_1,arg127_1,arg128_1,arg129_1,arg130_1,arg131_1,arg132_1,arg133_1,arg134_1,arg135_1,arg136_1,arg137_1,arg138_1,arg139_1,arg140_1,arg141_1,arg142_1,arg143_1,arg144_1,arg145_1,arg146_1,arg147_1,arg148_1,arg149_1,arg150_1,arg151_1,arg152_1,arg153_1,arg154_1,arg155_1,arg156_1,arg157_1,arg158_1,arg159_1,arg160_1,arg161_1,arg162_1,arg163_1,arg164_1,arg165_1,arg166_1,arg167_1,arg168_1,arg169_1,arg170_1,arg171_1,arg172_1,arg173_1,arg174_1,arg175_1,arg176_1,arg177_1,arg178_1,arg179_1,arg180_1,arg181_1,arg182_1,arg183_1,arg184_1,arg185_1,arg186_1,arg187_1,arg188_1,arg189_1,arg190_1,arg191_1,arg192_1,arg193_1,arg194_1,arg196_1,arg197_1,arg200_1,arg201_1,arg202_1,arg205_1,arg206_1,arg207_1,arg208_1,arg209_1,arg210_1,arg211_1,arg212_1,arg213_1,arg214_1,arg215_1,arg216_1,arg217_1,arg218_1,arg219_1,arg220_1,arg221_1,arg222_1,arg223_1,arg224_1,arg225_1,arg226_1,arg227_1,arg228_1,arg229_1,arg230_1,arg231_1,arg232_1,arg233_1,arg234_1,arg235_1,arg236_1,arg237_1,arg238_1,arg239_1,arg240_1,arg241_1,arg242_1,arg243_1,arg244_1,arg245_1,arg246_1,arg247_1,arg248_1,arg249_1,arg250_1,arg251_1,arg252_1,arg253_1,arg254_1,arg255_1,arg256_1,arg257_1,arg258_1,arg259_1,arg260_1,arg261_1,arg262_1,arg263_1,arg264_1,arg265_1,arg266_1,arg267_1,arg268_1,rope_seqlen_default]
    outputs = [rms_norm_64]

    kernel_cpp_0(inputs, outputs, param)
    del arg0_1
    del arg1_1
    del arg2_1
    del arg3_1
    del arg4_1
    del arg5_1
    del arg6_1
    del arg7_1
    del arg8_1
    del arg9_1
    del arg10_1
    del arg11_1
    del arg12_1
    del arg13_1
    del arg14_1
    del arg15_1
    del arg16_1
    del arg17_1
    del arg18_1
    del arg19_1
    del arg20_1
    del arg21_1
    del arg22_1
    del arg23_1
    del arg24_1
    del arg25_1
    del arg26_1
    del arg27_1
    del arg28_1
    del arg29_1
    del arg30_1
    del arg31_1
    del arg32_1
    del arg33_1
    del arg34_1
    del arg35_1
    del arg36_1
    del arg37_1
    del arg38_1
    del arg39_1
    del arg40_1
    del arg41_1
    del arg42_1
    del arg43_1
    del arg44_1
    del arg45_1
    del arg46_1
    del arg47_1
    del arg48_1
    del arg49_1
    del arg50_1
    del arg51_1
    del arg52_1
    del arg53_1
    del arg54_1
    del arg55_1
    del arg56_1
    del arg57_1
    del arg58_1
    del arg59_1
    del arg60_1
    del arg61_1
    del arg62_1
    del arg63_1
    del arg64_1
    del arg65_1
    del arg66_1
    del arg67_1
    del arg68_1
    del arg69_1
    del arg70_1
    del arg71_1
    del arg72_1
    del arg73_1
    del arg74_1
    del arg75_1
    del arg76_1
    del arg77_1
    del arg78_1
    del arg79_1
    del arg80_1
    del arg81_1
    del arg82_1
    del arg83_1
    del arg84_1
    del arg85_1
    del arg86_1
    del arg87_1
    del arg88_1
    del arg89_1
    del arg90_1
    del arg91_1
    del arg92_1
    del arg93_1
    del arg94_1
    del arg95_1
    del arg96_1
    del arg97_1
    del arg98_1
    del arg99_1
    del arg100_1
    del arg101_1
    del arg102_1
    del arg103_1
    del arg104_1
    del arg105_1
    del arg106_1
    del arg107_1
    del arg108_1
    del arg109_1
    del arg110_1
    del arg111_1
    del arg112_1
    del arg113_1
    del arg114_1
    del arg115_1
    del arg116_1
    del arg117_1
    del arg118_1
    del arg119_1
    del arg120_1
    del arg121_1
    del arg122_1
    del arg123_1
    del arg124_1
    del arg125_1
    del arg126_1
    del arg127_1
    del arg128_1
    del arg129_1
    del arg130_1
    del arg131_1
    del arg132_1
    del arg133_1
    del arg134_1
    del arg135_1
    del arg136_1
    del arg137_1
    del arg138_1
    del arg139_1
    del arg140_1
    del arg141_1
    del arg142_1
    del arg143_1
    del arg144_1
    del arg145_1
    del arg146_1
    del arg147_1
    del arg148_1
    del arg149_1
    del arg150_1
    del arg151_1
    del arg152_1
    del arg153_1
    del arg154_1
    del arg155_1
    del arg156_1
    del arg157_1
    del arg158_1
    del arg159_1
    del arg160_1
    del arg161_1
    del arg162_1
    del arg163_1
    del arg164_1
    del arg165_1
    del arg166_1
    del arg167_1
    del arg168_1
    del arg169_1
    del arg170_1
    del arg171_1
    del arg172_1
    del arg173_1
    del arg174_1
    del arg175_1
    del arg176_1
    del arg177_1
    del arg178_1
    del arg179_1
    del arg180_1
    del arg181_1
    del arg182_1
    del arg183_1
    del arg184_1
    del arg185_1
    del arg186_1
    del arg187_1
    del arg188_1
    del arg189_1
    del arg190_1
    del arg191_1
    del arg192_1
    del arg193_1
    del arg194_1
    del arg195_1
    del arg196_1
    del arg197_1
    del arg198_1
    del arg199_1
    del arg200_1
    del arg201_1
    del arg202_1
    del arg203_1
    del arg204_1
    del arg205_1
    del arg206_1
    del arg207_1
    del arg208_1
    del arg209_1
    del arg210_1
    del arg211_1
    del arg212_1
    del arg213_1
    del arg214_1
    del arg215_1
    del arg216_1
    del arg217_1
    del arg218_1
    del arg219_1
    del arg220_1
    del arg221_1
    del arg222_1
    del arg223_1
    del arg224_1
    del arg225_1
    del arg226_1
    del arg227_1
    del arg228_1
    del arg229_1
    del arg230_1
    del arg231_1
    del arg232_1
    del arg233_1
    del arg234_1
    del arg235_1
    del arg236_1
    del arg237_1
    del arg238_1
    del arg239_1
    del arg240_1
    del arg241_1
    del arg242_1
    del arg243_1
    del arg244_1
    del arg245_1
    del arg246_1
    del arg247_1
    del arg248_1
    del arg249_1
    del arg250_1
    del arg251_1
    del arg252_1
    del arg253_1
    del arg254_1
    del arg255_1
    del arg256_1
    del arg257_1
    del arg258_1
    del arg259_1
    del arg260_1
    del arg261_1
    del arg262_1
    del arg263_1
    del arg264_1
    del arg265_1
    del arg266_1
    del arg267_1
    del arg268_1
    return (rms_norm_64)