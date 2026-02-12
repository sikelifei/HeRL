<div align="center">


<h1 align="center">
Experience Is the Best Teacher:
Motivating Effective Exploration in Reinforcement Learning for LLMs
</h1>
<p align="center"><em>
HeRL: A hindsight-experience-guided reinforcement learning framework that bootstraps effective exploration by explicitly conveying the reward-specified target behaviors to LLMs.
</em></p>

---

HeRL is a hindsight-experience-guided RL framework built on top of veRL. The core changes are in
`verl/trainer/ppo/ray_trainer.py` and `verl/trainer/ppo/core_algos.py`, and the training scripts
live under `examples/`.

# ✨Getting Started

## Installation

You can install HeRL dependencies by running the following commands:

first download docker images:

```bash
docker pull hiyouga/verl:ngc-th2.5.1-cu120-vllm0.7.4-hotfix
docker run -it \
  --gpus all \
  --name your_name \
  --shm-size=64g \
  -v /your_path/herl:/workspace/herl \
  hiyouga/verl:ngc-th2.5.1-cu120-vllm0.7.4-hotfix \
  /bin/bash
pip install -r requirements.txt
```

## Repo Structure

This repository includes:

- `verl/`: HeRL implementation (core changes in `verl/trainer/ppo/ray_trainer.py` and
  `verl/trainer/ppo/core_algos.py`).
- `verl/examples/`: Training/launch scripts.

HeRL is built on top of veRL; most other folders keep the upstream layout unchanged.

# 🔧Usage

## Training

We provide an example script to train HeRL. You can run the following command to train HeRL:

```bash
  PYTHONPATH=/base_dir/verl sh /base_dir/verl/examples/
```


# Acknowledgement

HeRL builds upon [veRL](https://github.com/volcengine/verl). 

# Contact

For questions, feedback, feel free to reach out:
- Wenjian Zhang:zhangwenj@mail.dlut.edu.cn

# Citation
If you find our model, data, or evaluation code useful, please kindly cite our paper.
