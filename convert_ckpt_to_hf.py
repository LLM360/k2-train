import os
import fire
import json
import torch
import copy
from transformers import AutoConfig, LlamaForCausalLM
from transformers.modeling_utils import no_init_weights


HF_MODEL_NAME = "huggyllama/llama-65b"
TENSOR_PARALLEL = 8
PIPELINE_PARALLEL = 4
N_HEADS = 64
VOCAB_SIZE = 32032
# TEST_DATA_FILENAME = '/mount/data/shuffled_data_chunks/chunk_0.jsonl'


def load_from_chunks(models, param_name, src, dim, target_param_name, sd2load):
    if param_name == 'embedding':
        cat_weights = torch.cat([models[t]['embedding']['word_embeddings']['weight']
                    for t in range(TENSOR_PARALLEL)], dim=dim)
        sd2load[target_param_name] = cat_weights
    elif param_name == 'output_layer':
        cat_weights = torch.cat([models[t]['output_layer']['weight']
                    for t in range(TENSOR_PARALLEL)], dim=dim)
        sd2load[target_param_name] = cat_weights
    else:
        if isinstance(src, list):
            # chunks=2 for two components in the gated MLP layer
            chunks = [torch.chunk(models[t]['encoder'][param_name],
                chunks=2, dim=dim) for t in range(TENSOR_PARALLEL)]
            chunks = [torch.cat(c, dim=dim) for c in zip(*chunks)]
            for tpn, _c in zip(target_param_name, chunks):
                assert sd2load[tpn].size() == _c.size()
                sd2load[tpn] = _c
        elif dim == -1:
            sd2load[target_param_name] = models[0]['encoder'][param_name]
        else:
            if isinstance(target_param_name, list):
                #handle qkv:
                h1, h2 = src.shape
                reshaped_weights = torch.cat([
                    models[t]['encoder'][param_name]
                    for t in range(TENSOR_PARALLEL)
                ], dim=0).view(N_HEADS, -1, h2)
                chunked_reshaped_weights = torch.chunk(
                    reshaped_weights, chunks=3, dim=1) # 3 for qkv
                for tpn, crw in zip(
                        target_param_name, chunked_reshaped_weights):
                    crw = crw.contiguous().view(-1, h2)
                    assert sd2load[tpn].size() == crw.size()
                    sd2load[tpn] = crw
            else:
                #handle attn.o_proj:
                cat_weights = torch.cat([models[t]['encoder'][param_name]
                    for t in range(TENSOR_PARALLEL)], dim=dim)
                sd2load[target_param_name] = cat_weights


def main(load_path='/mount/ckpts/llama-65b-mp/iter_0096923',
         save_path='/mount/ckpts/65b_ckpts_hf/iter_0096923'):
    with no_init_weights():
        model = LlamaForCausalLM(
            config=AutoConfig.from_pretrained(HF_MODEL_NAME, vocab_size=VOCAB_SIZE))
    hf_state_dict = model.state_dict()

    ret = [
        [
            {
                'model': {'language_model': {'encoder': {}}},
                'checkpoint_version': 2
            } for _ in range(TENSOR_PARALLEL)
        ] for _ in range(PIPELINE_PARALLEL)
    ]

    for i in range(PIPELINE_PARALLEL):
        for j in range(TENSOR_PARALLEL):
            shard_name = f'mp_rank_{j:02d}_{i:03d}/'
            print(f'loading {os.path.join(load_path, shard_name)}')

#             os.makedirs(os.path.join(load_path, shard_name), exist_ok=True)
            ret[i][j] = torch.load(
                os.path.join(load_path, shard_name, 'model_optim_rng.pt'),
                map_location=torch.device('cpu')
            )['model']['language_model']

    new_state_dict = copy.deepcopy(model.state_dict())

    total = (len(hf_state_dict) - 3) // 9
    step = total // PIPELINE_PARALLEL
    # i: PP dim index
    # j: encoder block index
    # k: encoder block index per PP dim
    for i in range(PIPELINE_PARALLEL):
        end = total if i == PIPELINE_PARALLEL - 1 else (i + 1) * step
        for j in range(i * step, end):
            k = j - i * step

            load_from_chunks(
                ret[i],
                param_name=f'layers.{k}.input_layernorm.weight',
                src=hf_state_dict[f'model.layers.{j}.input_layernorm.weight'],
                dim=-1,
                target_param_name=f'model.layers.{j}.input_layernorm.weight',
                sd2load=new_state_dict)

            load_from_chunks(
                ret[i],
                param_name=f'layers.{k}.self_attention.query_key_value.weight',
                src=hf_state_dict[f'model.layers.{j}.self_attn.q_proj.weight'],
                dim=0,
                target_param_name=[
                 f'model.layers.{j}.self_attn.q_proj.weight',
                 f'model.layers.{j}.self_attn.k_proj.weight',
                 f'model.layers.{j}.self_attn.v_proj.weight'],
                sd2load=new_state_dict)

            load_from_chunks(
                ret[i],
                param_name=f'layers.{k}.self_attention.dense.weight',
                src=hf_state_dict[f'model.layers.{j}.self_attn.o_proj.weight'],
                dim=1,
                target_param_name=f'model.layers.{j}.self_attn.o_proj.weight',
                sd2load=new_state_dict)

            load_from_chunks(
                ret[i],
                param_name=f'layers.{k}.post_attention_layernorm.weight',
                src=hf_state_dict[f'model.layers.{j}.post_attention_layernorm.weight'],
                dim=-1,
                target_param_name=f'model.layers.{j}.post_attention_layernorm.weight',
                sd2load=new_state_dict)

            load_from_chunks(
                ret[i],
                param_name=f'layers.{k}.mlp.dense_h_to_4h.weight',
                src=[
                    hf_state_dict[f'model.layers.{j}.mlp.gate_proj.weight'],
                    hf_state_dict[f'model.layers.{j}.mlp.up_proj.weight'],
                ],
                dim=0,
                target_param_name=[
                    f'model.layers.{j}.mlp.gate_proj.weight',
                    f'model.layers.{j}.mlp.up_proj.weight'],
                sd2load=new_state_dict)

            load_from_chunks(
                ret[i],
                param_name=f'layers.{k}.mlp.dense_4h_to_h.weight',
                src=hf_state_dict[f'model.layers.{j}.mlp.down_proj.weight'],
                dim=1,
                target_param_name=f'model.layers.{j}.mlp.down_proj.weight',
                sd2load=new_state_dict)

    load_from_chunks(
        ret[0],
        param_name='embedding',
        src=hf_state_dict['model.embed_tokens.weight'],
        dim=0,
        target_param_name='model.embed_tokens.weight',
        sd2load=new_state_dict)

    load_from_chunks(
        ret[-1],
        param_name='final_layernorm.weight',
        src=hf_state_dict['model.norm.weight'],
        dim=-1,
        target_param_name='model.norm.weight',
        sd2load=new_state_dict)

    load_from_chunks(
        ret[-1],
        param_name='output_layer',
        src=hf_state_dict['lm_head.weight'],
        dim=0,
        target_param_name='lm_head.weight',
        sd2load=new_state_dict)

    model.load_state_dict(new_state_dict)
    model.save_pretrained(save_path, safe_serialization=False)
    print("Converting to HF Done !")

#     token_ids = json.loads(open(TEST_DATA_FILENAME).readline())['token_ids']
#     input_ids = torch.tensor([token_ids])
#     labels = torch.tensor([token_ids])

#     model.eval()
#     output_recons = model(input_ids, labels=labels, output_hidden_states=True)
#     print("### recons loss: {}".format(output_recons.loss))


if __name__ == '__main__':
    fire.Fire(main)