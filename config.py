from easydict import EasyDict as edict

use_tf = False
multi_thread = False

shape = 5
lr = 1.5e-2
beta = 1
reward_type = 'E'
approx_k = 200

beta_select = 10000
K = 1
train_iter = 30 

mode = 'omni'

noise_scale_min = 0
noise_scale_max = 0.3
noise_scale_decay = 300

config_T = edict({'shape': shape, 'approx_type': 'gsm', 'beta': beta, 'shuffle_state_feat': False,
                  'lr': lr, 'sample_size': 10, 'use_tf': use_tf, 'approx_k': approx_k, 'beta_select': beta_select})
config_L = edict({'shape': shape, 'approx_type': 'gsm', 'beta': beta, 'lr': lr, "prob": 1,
                  'shuffle_state_feat': mode == 'imit', 'particle_num': 1, 'replace_count': 1,
                  'noise_scale_min': noise_scale_min, 'noise_scale_max': noise_scale_max, 'noise_scale_decay': noise_scale_decay, 'cont_K': K,
                  'target_ratio': 0, 'new_ratio': 1, 'use_tf': use_tf, 'approx_k': approx_k, 'beta_select': beta_select})
