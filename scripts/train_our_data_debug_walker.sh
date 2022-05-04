GPU=$1
ENV=$2
DATANAME=$3
RANK=$4
ITER=$5

CUDA_VISIBLE_DEVICES=${GPU} python run_our_env.py --env ${ENV} --dataname ${DATANAME} --r ${RANK} \
--debug --iterations ${ITER} --adapt_iterations ${ITER} --num_simulators 1 --config exp_config/walker_rand_params.json --num_mpc_evals 1