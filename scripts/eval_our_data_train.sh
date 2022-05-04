GPU=$1
ENV=$2
DATANAME=$3
RANK=$4
NUM_SIM=$5

CUDA_VISIBLE_DEVICES=${GPU} python run_our_env.py --env ${ENV} --dataname ${DATANAME} --r ${RANK} \
--num_simulators ${NUM_SIM} --skip_train --eval_train --adapt_iterations 1