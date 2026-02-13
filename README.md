<h1 align="center">
Experience Is the Best Teacher:
Motivating Effective Exploration in Reinforcement Learning for LLMs
</h1>
<p align="center"><em>
HeRL: A hindsight-experience-guided reinforcement learning framework that bootstraps effective exploration by explicitly conveying the reward-specified target behaviors to LLMs.
</em></p>
<hr>



![framework](.\figures\framework.png)

# Getting Started

## Installation

You can install HeRL dependencies by running the following commands:

first download docker images:

```bash
docker pull hiyouga/verl:ngc-th2.5.1-cu120-vllm0.7.4-hotfix
docker run -it \
  --gpus all \
  --name your_name \
  --shm-size=64g \
  -v /your_path/HeRL:/workspace/herl \
  hiyouga/verl:ngc-th2.5.1-cu120-vllm0.7.4-hotfix \
  /bin/bash
pip install -r requirements.txt
nltk.download("punkt")
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
```

## Dataset

We use two datasets in this project:

1. **HIR-16K**([Anonymized Repository - Anonymous GitHub](https://anonymous.4open.science/r/HIR_Code-ECF2/README.md))
2. **RaR-Medicine**([anisha2102/RaR-Medicine · Datasets at Hugging Face](https://huggingface.co/datasets/anisha2102/RaR-Medicine))

## Repo Structure

This repository includes:

- `verl/`: HeRL implementation (core changes in `verl/trainer/ppo/ray_trainer.py` and
  `verl/trainer/ppo/core_algos.py`).
- `HeRL/examples/`: Training/launch scripts.

HeRL is built on top of veRL; most other folders keep the upstream layout unchanged.

# Usage

## Training

We provide example launch scripts under `examples/`.  
To train HeRL, run your training script with the project root in `PYTHONPATH`:

```bash
PYTHONPATH=/path_to_HeRL sh /path_to_HeRL/verl/examples/your_train_script.sh
```

# Result

HeRL is evaluated across **three base models** and **six benchmarks** under a unified training setup.  
Compared with standard post-training baselines (**SFT**, **DPO**, **RLVR**), HeRL consistently achieves the best overall performance, while also improving out-of-training-domain generalization (e.g., **WritingBench**).

## Main Results

> Metrics are reported in **accuracy (%)**.  
> Bold indicates the best score within each model block.

### Qwen2.5-7B-Instruct

| Method              |   IFEval |  IFBench | MulDimIF | WritingBench* | LLMEval-Med | HealthBench-500 |
| ------------------- | -------: | -------: | -------: | ------------: | ----------: | --------------: |
| Qwen2.5-7B-Instruct |     72.6 |     26.2 |     51.4 |          57.0 |        56.0 |            24.4 |
| + SFT               |     75.6 |     27.9 |     67.8 |          51.5 |        34.8 |            27.2 |
| + DPO               |     66.9 |     25.9 |     56.5 |          52.1 |        35.8 |            28.0 |
| + RLVR              |     77.3 |     31.6 |     73.5 |          54.8 |        60.5 |            30.5 |
| **+ HeRL (Ours)**   | **82.4** | **39.7** | **83.4** |      **59.1** |    **65.0** |        **34.3** |

### Llama-3.2-3B-Instruct

| Method                |   IFEval |  IFBench | MulDimIF | WritingBench* | LLMEval-Med | HealthBench-500 |
| --------------------- | -------: | -------: | -------: | ------------: | ----------: | --------------: |
| Llama-3.2-3B-Instruct |     71.2 |     23.8 |     35.8 |          30.5 |        16.1 |            14.5 |
| + SFT                 |     73.0 |     24.8 |     66.9 |          24.5 |        15.1 |            21.0 |
| + DPO                 |     74.3 |     22.1 |     54.4 |          14.4 |        11.5 |            13.5 |
| + RLVR                |     79.1 |     26.6 |     77.6 |          39.7 |        18.5 |            17.8 |
| **+ HeRL (Ours)**     | **82.4** | **30.6** | **84.7** |      **45.4** |    **18.7** |        **26.6** |

### Qwen3-4B-Instruct-2507

| Method                 |   IFEval |  IFBench | MulDimIF | WritingBench* | LLMEval-Med | HealthBench-500 |
| ---------------------- | -------: | -------: | -------: | ------------: | ----------: | --------------: |
| Qwen3-4B-Instruct-2507 |     83.4 |     29.9 |     57.3 |          84.3 |        74.5 |            42.0 |
| + SFT                  |     83.4 |     31.3 |     66.8 |          81.7 |        73.3 |            36.0 |
| + DPO                  |     83.9 |     27.9 |     61.5 |          85.0 |        74.9 |            39.1 |
| + RLVR                 |     85.8 |     36.9 |     79.0 |          83.9 |        78.1 |            41.7 |
| **+ HeRL (Ours)**      | **86.1** | **37.7** | **82.5** |      **85.7** |    **79.3** |        **43.6** |

---

## Ablation Study

We conduct ablations to isolate the contribution of each component in HeRL.  
**NaiveHE** denotes naive hindsight experience construction, **HE** denotes the full hindsight experience design, and **BR** denotes bonus reward shaping.  
Values are accuracy (%), and numbers in parentheses indicate gains over the RLVR baseline.

### Model I: Qwen2.5-7B-Instruct

| Setting              |         IFBench |    WritingBench | HealthBench-500 |
| -------------------- | --------------: | --------------: | --------------: |
| Baseline (RLVR)      |     31.6 (+0.0) |     54.8 (+0.0) |     30.5 (+0.0) |
| + NaiveHE            | 32.3 (**+0.7**) | 52.7 (**-2.1**) | 28.8 (**-1.7**) |
| + HE                 | 36.7 (**+5.1**) | 58.9 (**+4.1**) | 31.8 (**+1.3**) |
| **+ HE + BR (HeRL)** | **39.7 (+8.1)** | **59.1 (+4.3)** | **34.3 (+3.8)** |

### Model II: Llama-3.2-3B-Instruct

| Setting              |         IFBench |    WritingBench | HealthBench-500 |
| -------------------- | --------------: | --------------: | --------------: |
| Baseline (RLVR)      |     26.6 (+0.0) |     39.7 (+0.0) |     17.8 (+0.0) |
| + NaiveHE            | 28.3 (**+1.7**) | 33.7 (**-6.0**) | 23.9 (**+6.1**) |
| + HE                 | 28.1 (**+1.5**) | 41.1 (**+1.4**) | 23.5 (**+5.7**) |
| **+ HE + BR (HeRL)** | **30.6 (+4.0)** | **45.4 (+5.7)** | **26.6 (+8.8)** |

# Acknowledgement

HeRL builds upon [veRL](https://github.com/volcengine/verl). 

# Contact

For questions, feedback, feel free to reach out:
- Wenjian Zhang:zhangwenj@mail.dlut.edu.cn

# Citation
If you find our model, data, or evaluation code useful, please kindly cite our paper.
