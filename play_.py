import numpy as np
import copy
import os
import sys

from map import Map
from teacher_irl import TeacherIRL
from learner_irl import LearnerIRL
from learnfromhuman import LearnFromHuman
exec('import config', globals())
exec('from config import config_T', globals())
exec('from config import config_L', globals())

def run():
    mode = config.mode

    train_iter = config.train_iter

    seed = 1
    np.set_printoptions(precision = 4)

    map_num = 6
    feedback = True

    np.random.seed((seed + 1) * 157)

    map_l = Map(config_L)
    np.random.seed((seed + 1) * 163)

    map_t = Map(config_T)
    np.random.seed((seed + 1) * 174)

    gt_r_param_tea = map_l.reward_grid(map_num) #map_l.reward_generate(3) if config.reward_type == 'E' else np.random.uniform(-2, 2, size = [1, config.shape ** 2])
    gt_r_param_stu = copy.deepcopy(gt_r_param_tea)
    if config_L.shuffle_state_feat:
        #print("Shuffling with ", map_l.feat_idx_)
        gt_r_param_stu[:, map_l.feat_idx_] = gt_r_param_tea

    assert(np.max(abs(np.sum(gt_r_param_stu * map_l.state_feats_, axis = 1) - np.sum(gt_r_param_tea * map_t.state_feats_, axis = 1))) < 1e-9)

    np.random.seed((seed + 1) * 105)
    teacher = TeacherIRL(map_t, config_T, gt_r_param_tea, gt_r_param_stu)
    init_ws = np.random.uniform(-2, 2, size = [config_L.particle_num, teacher.map_.num_states_])
    unshuffled_ws = copy.deepcopy(init_ws)
    test_set = np.random.choice(teacher.map_.num_states_, size = [train_iter + 1, teacher.map_.num_states_ * 20])
    learner = LearnerIRL(map_l, config_L)
    lfh = LearnFromHuman(teacher, learner, init_ws, test_set, gt_r_param_tea)
    return lfh