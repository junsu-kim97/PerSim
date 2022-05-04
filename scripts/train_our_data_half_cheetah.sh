GPU=$1
ENV=$2
DATANAME=$3
RANK=$4
ITER=$5
ADAPT_ITER=$6
SEED=$7

CUDA_VISIBLE_DEVICES=${GPU} python run_our_env.py --env ${ENV} --dataname ${DATANAME} --r ${RANK} \
--iterations ${ITER} --adapt_iterations ${ADAPT_ITER} --config exp_config/half_cheetah_rand_params.json \
--num_simulators 5 --num_mpc_evals 1 --seed ${SEED}