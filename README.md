# Behaviour Policy Optimization (BPO)

The repository provide the official implementation of **Behaviour Policy Optimization (BPO)**, an off-policy extension of the classic **Proximal Policy Optimization (PPO)** algorithm with provable variance reduction. The full details can be found in:

> **Behaviour Policy Optimization: Provably Lower Variance Return Estimates for Off-Policy Reinforcement Learning**
> Alexander W. Goodall, Edwin Hamel-De le Court, Francesco Belardinelli
> arXiv: [https://arxiv.org/abs/2511.10843](https://arxiv.org/abs/2511.10843) 

*Disclaimer:* This implementation is simplified from the full version used to collect the results provided in the paper.

## Quick Start

### Installation

#### Prequisites

Python 3.8+ is required but we recommend Python 3.10 (later Python versions may not be supported).

#### Installation with conda 

#### Installation with conda 
- Install conda, e.g., via [anaconda](https://anaconda.org/channels/anaconda/packages/conda/overview).
- Clone the repo:
```bash
git clone https://github.com/sacktock/BPO.git
cd BPO
```
- Create a conda virtual environment:
```bash
conda env create --name bpo --file conda-environment.yaml
conda activate bpo
```
- Install dependencies:
```bash
pip install -r requirements.txt
```

### Check the installation is working

```
python run.py --configs cartpole_ppo
```

### Enabling GPU Acceleration with JAX (Optional)

Our implementation relies on [JAX](https://docs.jax.dev/) for GPU acceleration, which can be enabled for Linux or WSL sub systems. 

- **Linux x86_64/aarch64**: jax and jaxlib `0.4.30` should already be installed via the `requirements.txt`. You need to reinstall JAX based on your cuda driver compatibility. Do not use the ```-U``` option here!
```bash
pip install "jax[cuda12]"
```
For `13+` cuda versions you may need to upgrade the jax and jaxlib installation. 

- **Windows**: GPU acceletartion is also supported (experimentally) on Windows WSL x86_64. We strongly recommend using [Ubuntu 22.04](https://apps.microsoft.com/detail/9pn20msr04dw?hl=en-GB&gl=BE) or similar. Follow the **Linux x86_64/aarch64** instructions above. 

- **MAC**: we recommend JAX with CPU. No further action is required if you correctly followed the earlier steps.

## Reproducing our Results

All experiments are launched from the command line via `run.py`. Under the hood, `run.py` loads **named configuration blocks** from `configs.yaml` and merges them **left -> right** in the order you pass them to `--configs`. Later configs override earlier ones.  

### Running experiments

The basic pattern is:

```bash
python run.py --configs <base_config> [<override_config> ...] [flag overrides...]
```

* `--configs` can take **one or more** names, each corresponding to a top-level key in `configs.yaml` (e.g., `cartpole_ppo`, `mujoco_ppo_gsde`, `ppo_bpo_zero`, ...). 
* After config merging, any remaining CLI arguments are parsed as **overrides** (e.g., `--env.env_id ant`, `--run.seed 0`, etc.). 

A minimal PPO run (CartPole preset):

```bash
python run.py --configs cartpole_ppo
```

This uses the `cartpole_ppo` block (timesteps, env, PPO hyperparameters, etc.). 
A MuJoCo PPO run (Ant baseline preset):

```bash
python run.py --configs mujoco_ppo --env.env_id ant --run.seed 0 --run.logdir runs/mujoco/ant/ppo_seed_0
```

(`mujoco_ppo` sets MuJoCo-style PPO defaults; `--env.env_id` can be changed among `ant`, `half_cheetah`, `hopper`, `walker_2d`.)  

### Enabling BPO

**Option A -> flip the flag directly**

```bash
python run.py --configs mujoco_ppo_gsde --bpo True --env.env_id ant --run.seed 0
```

**Option B -> include a BPO config block**
For example, `ppo_bpo_zero` is a small "add-on" config that sets `bpo: True` and applies BPO-specific settings (e.g., `symlog_targets`, `polyak_tau`, and a zero-norm-final Q-head). 

```bash
python run.py --configs mujoco_ppo_gsde ppo_bpo_zero --env.env_id ant --run.seed 0
```

### Switching BPO hyperparameter settings

The repo includes several ready-made BPO variants that mainly differ in **importance-weight clipping** (`clip_rho`, `clip_c`) and whether **trajectory clipping** is enabled (`clip_traj`). 

Common presets:

* `ppo_bpo_zero` -> `clip_rho=1.5`, `clip_c=1.5`, plus "zero final norm" Q-head 
* `ppo_bpo_zero1.0` -> `clip_rho=1.0`, `clip_c=1.0` 
* `ppo_bpo_zero1.0_1.5` -> `clip_rho=1.5`, `clip_c=1.0` 
* `ppo_bpo_zero1.0_1.4` -> `clip_rho=1.4`, `clip_c=1.0` 
* `ppo_bpo_zero1.0_traj` -> `clip_rho=1.0`, `clip_c=1.0`, `clip_traj=True` 
* `ppo_bpo_zero1.5_traj` -> `clip_rho=1.5`, `clip_c=1.5`, `clip_traj=True` 

So, to run the "rho/c = 1.0/1.0" setting you can do:

```bash
python run.py --configs mujoco_ppo_gsde ppo_bpo_zero1.0 --env.env_id ant --run.seed 0
```

### Logging and Miscellaneous

**TensorBoard** logging is supported by our implementation and is enabled in the following example:

```bash
python run.py \
  --configs mujoco_ppo_gsde ppo_bpo_zero \
  --env.env_id ant \
  --tensorboard True \
  --logdir "runs/mujoco/ant/ppo_gsde_fqe_zero_seed_0" \
  --seed 0
```

The TensorBoard logs can be accessed via the command line:

```bash
tensorboard --logdir runs/mujoco/ant/
```

The `verbose` flag controls **how much diagnostic information is logged during training**. This affects both **console output** and the **set of metrics written to TensorBoard / logs**. There are **three verbosity levels**, with default set to `0`.

The verbosity level can be changed via a command-line override:

```bash
--run.verbose <level>
```

SImilarly, you can also override any nested field directly from the command line (after configs are merged). For example, to keep `ppo_bpo_zero` but change the clipping thresholds:

```bash
python run.py --configs mujoco_ppo_gsde ppo_bpo_zero \
  --env.env_id ant \
  --ppo.clip_rho 1.0 --ppo.clip_c 1.0 \
  --run.seed 0
```



## License

Out implementation of BPO is released under the MIT License.



