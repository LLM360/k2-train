# if on FM cluster, please do NOT do these steps on the login node
# successfully tested on the FM cluster by hongyi on aug 8, 2023

# step 0 create conda venv
conda create -n megatron-llm python=3.9
conda activate megatron-llm

# step 1 install pytorch
module load cuda/12.1
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121

pip uninstall -y ninja && pip install ninja
pip install packaging numpy six regex transformers datasets pybind11

# [DO NOT run on login node] APEX installation, has to be done on a GPU node 
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

# [DO NOT run on login node] install flashattention, please run on a CPU or GPU node
MAX_JOBS=1 pip install flash-attn --no-build-isolation