# ROMI: Reverse Offline Model-based Imagination

This repository is the implementation of the paper [Offline Reinforcement Learning with Reverse Model-based Imagination](https://arxiv.org/abs/2110.00188).

## Setup the environment

```
conda create -n romi python=3.6.5
pip install -r environment/requirements.txt
pip install -e ./mopo
pip install -e ./CQL/d4rl
```

To run the code, you can configure the environment in bash examples in `bash/`. Before running the code, set current directory as PYTHONPATH in examples.

## To learn reverse models
```bash
cd bash
bash train_reverse_model.sh
```

## To train diverse rollout policy
```bash
cd bash
bash train_reverse_bc.sh
```

## To train ROMI-BCQ
```bash
cd bash
bash train_romi_bcq.sh
```

## To train ROMI-CQL
```bash
cd bash
bash train_romi_cql.sh
```