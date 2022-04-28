from envs import CartPoleEnv, MountainCarEnv, HalfCheetahEnv
from rlkit.envs import ENVS

def get_environment_config(env):
    config = {"env":env}
    
    if config['env'] == 'cartPole':
        train_force_set = [2.0, 18] # for testing choose 2, 10,18
        train_length_set = [0.15, 0.85] # for testing choose 0.15, 0.5 ,0.85
        config['covariates'] = ['force', 'length']
        config['env_default'] = [10,0.5]
        config['train_env_range'] = [train_force_set, train_length_set]
        config['test_env'] = [[2.0,0.5],[10.0,0.5],[18., 0.5],[10.0,0.85],[10., 0.15]]
        config['train_policy'] = [[6.0,0.5],[14.0,0.5],[10., 0.5],[10.0,0.675],[10., 0.325]]
        config['trajectory_length'] = 200
        config['no_trajectories'] = 1
        config['number_of_test_rollouts'] = 200
        config['number_of_units'] = 500
        config['env'] = CartPoleEnv
    
    elif config['env'] == 'mountainCar':
        train_gravity_set = [0.0001, 0.0035]
        config['covariates'] = ['gravity']
        config['env_default'] = [0.0025]
        config['train_env_range'] = [train_gravity_set]
        config['test_env'] = [[0.0001],[0.0005],[0.001],[0.0025],[0.0035]]
        config['train_policy'] = [[0.0003],[.00075],[0.00175],[0.0025],[0.003]]
        config['trajectory_length'] = 500
        config['number_of_test_rollouts'] = 200
        config['number_of_units'] = 500
        config['env'] = MountainCarEnv
        config['no_trajectories'] = 1
     
    elif config['env'] == 'halfCheetah':
        train_mass_scale_set = [0.2,1.8] # test 0.2, ,1.0, 1.8
        train_damping_scale_set = [0.2,1.8] # test 0.2, ,1.0, 1.8
        config['covariates'] = ['mass_scale_set', 'damping_scale_set']
        config['env_default'] = [1.0,1.0]
        config['train_env_range'] = [train_mass_scale_set, train_damping_scale_set]
        config['test_env'] =  [[0.3,1.7],[1.7,0.3], [0.3,0.3],[1.7,1.7],[1.,1.]]
        config['train_policy'] = [[1.4,0.6],[1.4,1.4], [0.6,0.6],[0.6,1.4]]
        config['trajectory_length'] = 1000
        config['number_of_test_rollouts'] = 200
        config['number_of_units'] = 500
        config['env'] = HalfCheetahEnv
        config['no_trajectories'] = 1

    # TODO: check hyperparameters for walker-rand-params
    elif config['env'] == 'walker-rand-params':
        train_mass_scale_set = [None, None] # test 0.2, ,1.0, 1.8
        train_damping_scale_set = [None, None] # test 0.2, ,1.0, 1.8
        config['covariates'] = [None, None]
        config['env_default'] = [None, None]
        config['train_env_range'] = [None, None]
        config['test_env'] =  [[None, None], [None, None], [None, None], [None, None], [None, None]]
        config['train_policy'] = [[None, None], [None, None], [None, None], [None, None]]
        config['trajectory_length'] = None
        config['number_of_test_rollouts'] = None
        config['number_of_train_units'] = 20
        config['number_of_test_units'] = 4

        config['number_of_train_idxs'] = 50
        config['number_of_test_adapt_idxs'] = 40
        config['number_of_test_pred_idxs'] = 10
        config['env'] = ENVS['walker-rand-params']
        config['no_trajectories'] = None
        
    else:
        raise ValueError('Choose from available envs {"slimHumanoid", "ant", "halfCheetah", "mountainCar", "cartPole", "walker-rand-params"}, refer to envs/config.py')


    return config
