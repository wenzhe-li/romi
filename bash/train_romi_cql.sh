#!/bin/bash
cd ..
mkdir outlogs_romi_cql &> /dev/null
declare -a tasks=(
  # "maze2d-umaze-v1" "maze2d-medium-v1" "maze2d-large-v1" \
  # "maze2d-umaze-dense-v1" "maze2d-medium-dense-v1" "maze2d-large-dense-v1" \
  # "antmaze-umaze-v0" "antmaze-medium-play-v0" "antmaze-large-play-v0" \
  # "antmaze-umaze-diverse-v0" "antmaze-medium-diverse-v0" "antmaze-large-diverse-v0" \
  # "halfcheetah-medium-expert-v2" "halfcheetah-medium-replay-v2" "halfcheetah-random-v2" "halfcheetah-medium-v2" \
  # "walker2d-medium-expert-v2" "walker2d-medium-replay-v2" "walker2d-random-v2" "walker2d-medium-v2" \
  # "hopper-medium-expert-v2" "hopper-medium-replay-v2" "hopper-random-v2" "hopper-medium-v2" \
)
declare -a algos=( "romi_cql" )
declare -a seeds=( "1" )

export PYTHONPATH=""

n=3
gpunum=8
for task in "${tasks[@]}"
do
for algo in "${algos[@]}"
do
for seed in "${seeds[@]}"
do
nohup python CQL/d4rl/examples/cql_mujoco_new.py \
--env=${task} --seed=${seed} --gpu=${n} \
>& outlogs_romi_cql/${task}_${algo}_${seed}.log &
echo "task: ${task}, algo: ${algo}, seed: ${seed}, GPU: $n"
n=$[($n+1) % ${gpunum}]
sleep 1
done
done
done