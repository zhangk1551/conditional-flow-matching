# Mean Flows

## Install
Creat a conda environment and run `pip install -r requirements.txt`


## Running

**Configuration priority (from highest to lowest):**

1. Weights & Biases (wandb) sweep config 
2. wandb YAML config 
3. Default settings in `config.py`

### Run 8 Gaussians / Moons

**Example run**: [wandb link](https://wandb.ai/zhangk15/mean-flow/runs/n9fozdwy)

`python main.py --conf conf/gaussians_moons.yml --wandb_mode "online"`

**Example sweep**: [wandb link](https://wandb.ai/zhangk15/mean-flow/sweeps/fxgqeuph)


Start a new sweep via:

`wandb sweep --project mean-flow conf/gaussians_moons_sweep.yml`

Start a run under a sweep:
`python main.py --wandb_mode "online" --sweep_id $SWEEP_ID`


### Run MNIST

**Example run**: [wandb link](https://wandb.ai/zhangk15/mean-flow/runs/k387c0gu)


`python main.py --conf conf/mnist.yml --wandb_mode "online"`



## References
Paper: [Mean Flows for One-step Generative Modeling](https://arxiv.org/pdf/2505.13447)
- [Flow Matching Tutorial (8 Gaussians / Moons)](https://github.com/atong01/conditional-flow-matching/blob/main/examples/2D_tutorials/Flow_matching_tutorial.ipynb) 
- [Conditional Flow Matching on MNIST](https://github.com/atong01/conditional-flow-matching/blob/main/examples/images/mnist_example.ipynb)
