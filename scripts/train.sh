GPU=$1
ENV=$2
DATANAME=$3
RANK=$4
TIMECUT=$5

CUDA_VISIBLE_DEVICES=${GPU} python run.py --env ${ENV} --dataname ${DATANAME} --r ${RANK} --time_cut ${TIMECUT}