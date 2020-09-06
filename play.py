import numpy as np
import copy

from map import Map
from teacher_irl import TeacherIRL
from learner_irl import LearnerIRL
from learn_human import LearnHuman
from session import Session 

exec('import config', globals())
exec('from config import config_T', globals())
exec('from config import config_L', globals())

def run(map_num, intro = False):
    print('[Initializing, please wait ...]')
    mode = config.mode

    if intro:
        train_iter = 10
    else:
        train_iter = config.train_iter

    np.set_printoptions(precision = 4)

    map_l = Map(config_L)

    map_t = Map(config_T)

    gt_r_param_tea = map_l.reward_grid(map_num) #map_l.reward_generate(3) if config.reward_type == 'E' else np.random.uniform(-2, 2, size = [1, config.shape ** 2])
    gt_r_param_stu = copy.deepcopy(gt_r_param_tea)
    if config_L['shuffle_state_feat']:
        #print("Shuffling with ", map_l.feat_idx_)
        gt_r_param_stu[:, map_l.feat_idx_] = gt_r_param_tea

    assert(np.max(abs(np.sum(gt_r_param_stu * map_l.state_feats_, axis = 1) - np.sum(gt_r_param_tea * map_t.state_feats_, axis = 1))) < 1e-9)

    teacher = TeacherIRL(map_t, config_T, gt_r_param_tea, gt_r_param_stu)
    teacher2 = TeacherIRL(map_t, config_T, gt_r_param_tea, gt_r_param_stu)

    init_ws = np.random.uniform(-2, 2, size = [1, teacher.map_.num_states_])
    unshuffled_ws = copy.deepcopy(init_ws)
    test_set = np.random.choice(teacher.map_.num_states_, size = [train_iter + 1, teacher.map_.num_states_ * 20])
    

    learner = LearnerIRL(map_l, config_L)

    s = Session(map_num)
    lfh_ital = LearnHuman(teacher, learner, init_ws, test_set, gt_r_param_tea, train_iter, config.feedback, map_num, s)

    s2 = Session(map_num, True)
    lfh_imt = LearnHuman(teacher2, learner, init_ws, test_set, gt_r_param_tea, train_iter, config.feedback, map_num, s2, 1)
    
    print('[Finish initialization, please continue with next block]')
    return lfh_ital, lfh_imt