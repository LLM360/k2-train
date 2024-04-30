# PLEASE RUN UNDER THE MEGATRON-LLM FOLDER
# EXAMPLE1 (NO MP or PP) :python ./tools/ckpts_from_hf.py --save_path ./ckpts-llama-160m --n-head 12
# EXAMPLE2 :python ./tools/ckpts_from_hf.py --save_path ./llama-7b-from-hf-tensor8-pipeline2 --n-head 32 --tensor-parallel 8 --pipeline-parallel 2
import torch
from copy import deepcopy
import os
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = ArgumentParser()
#parser.add_argument('--hf_path', type=str)
parser.add_argument('--save_path', type=str, required=True)
parser.add_argument('--tensor-parallel', type=int, default=1)
parser.add_argument('--pipeline-parallel', type=int, default=1)
parser.add_argument('--n-head', type=int, required=True)
args = parser.parse_args()

#MODEL_PATH="JackFram/llama-160m"
MODEL_PATH="huggyllama/llama-7b"

#config = AutoConfig.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
hf = model.state_dict()

ret = [[{'model': {'language_model': {'encoder': {}}}, 'checkpoint_version': 2} for _ in range(args.tensor_parallel)] for _ in range(args.pipeline_parallel)]

def save_into_chunks(models, param_name, src, dim):
    if isinstance(src, list):
        chunks = [torch.chunk(s, chunks=args.tensor_parallel, dim=dim) for s in src]
        chunks = [torch.cat(c, dim=dim) for c in zip(*chunks)]
    elif dim == -1:
        chunks = [src for _ in range(args.tensor_parallel)]
    else:
        chunks = torch.chunk(src, chunks=args.tensor_parallel, dim=dim)
    for i, chunk in enumerate(chunks):
        if param_name == 'embedding':
            models[i]['model']['language_model']['embedding'] = {'word_embeddings': {'weight': chunk}}
        elif param_name == 'output_layer':
            models[i]['model']['language_model']['output_layer'] = {'weight': chunk}
        else:
            models[i]['model']['language_model']['encoder'][param_name] = chunk

save_into_chunks(ret[0], 'embedding', hf['model.embed_tokens.weight'], 0)
save_into_chunks(ret[-1], 'final_layernorm.weight', hf['model.norm.weight'], -1)
save_into_chunks(ret[-1], 'output_layer', hf['lm_head.weight'], 0)

total = (len(hf) - 3) // 10
step = total // args.pipeline_parallel
for i in range(args.pipeline_parallel):
    end = total if i == args.pipeline_parallel - 1 else (i + 1) * step
    for j in range(i * step, end):
        k = j - i * step
        save_into_chunks(ret[i], f'layers.{k}.input_layernorm.weight', hf[f'model.layers.{j}.input_layernorm.weight'], -1)
        h1, h2 = hf[f'model.layers.{j}.self_attn.q_proj.weight'].shape
        save_into_chunks(ret[i], f'layers.{k}.self_attention.query_key_value.weight', torch.cat([
            hf[f'model.layers.{j}.self_attn.q_proj.weight'].view(args.n_head, -1, h2),
            hf[f'model.layers.{j}.self_attn.k_proj.weight'].view(args.n_head, -1, h2),
            hf[f'model.layers.{j}.self_attn.v_proj.weight'].view(args.n_head, -1, h2),
        ], dim=1).view(-1, h2), 0)
        save_into_chunks(ret[i], f'layers.{k}.self_attention.dense.weight', hf[f'model.layers.{j}.self_attn.o_proj.weight'], 1)
        save_into_chunks(ret[i], f'layers.{k}.post_attention_layernorm.weight', hf[f'model.layers.{j}.post_attention_layernorm.weight'], -1)
        save_into_chunks(ret[i], f'layers.{k}.mlp.dense_h_to_4h.weight', [
            hf[f'model.layers.{j}.mlp.gate_proj.weight'],
            hf[f'model.layers.{j}.mlp.up_proj.weight'],
        ], 0)
        save_into_chunks(ret[i], f'layers.{k}.mlp.dense_4h_to_h.weight', hf[f'model.layers.{j}.mlp.down_proj.weight'], 1)



name = lambda i, j: f'mp_rank_{j:02d}_{i:03d}/' if args.pipeline_parallel > 1 else f'mp_rank_{i:02d}/'
for i in range(args.pipeline_parallel):
    for j in range(args.tensor_parallel):
        os.makedirs(os.path.join(args.save_path, 'release', name(i, j)), exist_ok=True)
        torch.save(ret[i][j], os.path.join(args.save_path, 'release', name(i, j), 'model_optim_rng.pt'))
with open(os.path.join(args.save_path, 'latest_checkpointed_iteration.txt'), 'w') as f:
    f.write('{}'.format('release'))

# For reference, old code supporting single GPU

# ret = {'model': {'language_model':
#         {'embedding': {'word_embeddings': {}},
#          'encoder': {},
#          'output_layer': {}
#         }
# }}

# ret['model']['language_model']['embedding']['word_embeddings']['weight'] = hf['model.embed_tokens.weight']
# ret['model']['language_model']['encoder']['final_layernorm.weight'] = hf['model.norm.weight']
# ret['model']['language_model']['output_layer']['weight'] = hf['lm_head.weight']

# for i in range((len(hf) - 3) // 10):
#     ret['model']['language_model']['encoder'][f'layers.{i}.input_layernorm.weight'] = hf[f'model.layers.{i}.input_layernorm.weight']
#     ret['model']['language_model']['encoder'][f'layers.{i}.self_attention.query_key_value.weight'] = torch.cat([
#         hf[f'model.layers.{i}.self_attn.q_proj.weight'],
#         hf[f'model.layers.{i}.self_attn.k_proj.weight'],
#         hf[f'model.layers.{i}.self_attn.v_proj.weight'],
#     ], dim=0)
#     ret['model']['language_model']['encoder'][f'layers.{i}.self_attention.dense.weight'] = hf[f'model.layers.{i}.self_attn.o_proj.weight']
#     ret['model']['language_model']['encoder'][f'layers.{i}.post_attention_layernorm.weight'] = hf[f'model.layers.{i}.post_attention_layernorm.weight']
#     # ret['model']['language_model']['encoder'][f'layers.{i}.mlp.gate_proj.weight'] = hf[f'model.layers.{i}.mlp.gate_proj.weight']
#     # ret['model']['language_model']['encoder'][f'layers.{i}.mlp.up_proj.weight'] = hf[f'model.layers.{i}.mlp.up_proj.weight']
#     # ret['model']['language_model']['encoder'][f'layers.{i}.mlp.down_proj.weight'] = hf[f'model.layers.{i}.mlp.down_proj.weight']
#     ret['model']['language_model']['encoder'][f'layers.{i}.mlp.dense_h_to_4h.weight'] = torch.cat([
#        hf[f'model.layers.{i}.mlp.gate_proj.weight'],
#        hf[f'model.layers.{i}.mlp.up_proj.weight'],
#     ], dim=0)
#     ret['model']['language_model']['encoder'][f'layers.{i}.mlp.dense_4h_to_h.weight'] = hf[f'model.layers.{i}.mlp.down_proj.weight']