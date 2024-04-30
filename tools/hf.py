import torch
import datasets

from transformers import AutoConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

config = AutoConfig.from_pretrained("JackFram/llama-160m")
model = AutoModelForCausalLM.from_pretrained("JackFram/llama-160m")

ds = datasets.load_dataset("json", data_files=["/home/ubuntu/Megatron-LLM/test.json"])["train"]
ds = ds.with_format("np")
ds = ds.rename_column("token_ids", "text")
train_ds = ds

print("train ds0 : {}".format(train_ds[0]['text'].shape))
inputs = train_ds[0:2]['text']  #[0:2048].reshape(1, 2048)
label = train_ds[0:2]['text']  #[1:].reshape(1, 2048)
print("inputs: {}".format(inputs))
inputs, label = torch.from_numpy(inputs), torch.from_numpy(label)
output = model(inputs, labels=label, output_hidden_states=True)
#print("output hidden states: {}, len: {}".format(output.hidden_states, len(output.hidden_states)))

#for hs_index, hs in enumerate(output.hidden_states):
#    print(hs_index, hs.size())
print(output.loss)
# print(output.hidden_states[0].size())
# print(output.hidden_states[0][0,0,:])