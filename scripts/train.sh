GPU=$1
ENV=$2
DATANAME=$3

CUDA_VISIBLE_DEVICES=${GPU} python run.py --env ${ENV} --dataname ${DATANAME}