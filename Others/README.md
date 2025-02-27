### NVIDIA H100 GPUs

#### Multicast Support
**References** ->
[NCCL-op128](https://github.com/NVIDIA/nccl/blob/80f6bda4378b99d99e82b4d76a633791cc45fef0/src/device/op128.h#L359) | [MSCCLPP-NVLS](https://github.com/microsoft/mscclpp/blob/b3992a8b298103171a209abbe0925cac4a7e06b8/python/mscclpp_benchmark/allreduce.cu#L833) | [NVBandwith-multicast](https://github.com/NVIDIA/nvbandwidth/blob/64d683772a4d75e392b22c43eb065d33f4adb41b/kernels.cu#L127) | [cuda-programming-guide-multicast](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=multicast#multicast-support)

### H100 vllm setup
```bash
source ~/.bashrc
conda create -n vllm python=3.12 -y
conda activate vllm
sudo apt-get install libmkl-dev -y; conda install mkl -y
pip3 install -r requirements-common.txt; pip3 install -r requirements-cuda.txt ; pip3 install -r requirements-dev.txt; pip3 install -e .; python3 setup.py develop;
huggingface-cli login # token
```

### NVBandwidth setup

```bash
sudo apt install libboost-program-options-dev
sudo ./debian_install.sh
# For Multinode
cmake -DMULTINODE=1 .
make
## To Run
./nvbandwidth
### Multinode
mpirun --allow-run-as-root --map-by ppr:4:node --bind-to core -np 8 --report-bindings -q -mca btl_tcp_if_include enP5p9s0 --hostfile /etc/nvidia-imex/nodes_config.cfg  ./nvbandwidth -p multinode
```