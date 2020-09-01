from multiprocessing import Process, Manager
import copy
import os
import sys, io
import numpy as np
from easydict import EasyDict as edict
from tqdm import tqdm
import matplotlib.pyplot as plt

import time

from map import Map
from teacher_irl import TeacherIRL
from learner_irl import LearnerIRL

import pdb

from game import Game

def blockPrint():
    text_trap = io.StringIO()
    sys.stdout = text_trap
    
def enablePrint():
    sys.stdout = sys.__stdout__

def learn(teacher, learner, mode, init_ws, train_iter, test_set, teacher_rewards, random_prob = None, human_teacher=True, feedback=False):
    learner.reset(init_ws)
    teaching_examples = []
    batches = []
    dists_ = [np.sqrt(np.sum(np.square(learner.current_mean_ - teacher.stu_gt_reward_param_)))]
    dists = [np.mean(np.max(abs(learner.current_action_prob() - teacher.action_probs_), axis = 1))]
    distsq = [np.mean(np.square(learner.q_map_ - teacher.q_map_))]
    actual_rewards = [teacher.map_.test_walk(teacher.reward_param_, learner.action_probs_, test_set[0], greedy = True)]
    ws = []
    learned_rewards = []
    policy = []

    for i in range(train_iter):
        teacher.sample()
        learned_rewards.append(copy.deepcopy(learner.current_mean_))
        batches.append([teacher.mini_batch_indices_, teacher.mini_batch_opt_acts_])
        if mode[0: 4] == 'omni':
            data_idx, gradients = teacher.choose(learner.current_mean_, learner.lr_, hard = True)
            teaching_examples.append([teacher.mini_batch_indices_[data_idx], teacher.mini_batch_opt_acts_[data_idx]])
            if human_teacher:
                if feedback:
                    g = Game(teacher_rewards, batches[-1], copy.deepcopy(learner.q_map_), copy.deepcopy(learner.current_mean_), data_idx)
                else:
                    g = Game(teacher_rewards, batches[-1], copy.deepcopy(learner.q_map_), copy.deepcopy(learner.current_mean_))

                g.display()
                data_idx = g.selected_idx_
                plt.close('all')

        elif mode[0: 4] == 'imit':
            stu_rewards = np.sum(learner.map_.state_feats_ * learner.current_mean_, axis = 1, keepdims = True)
            data_idx, gradients, l_stu = teacher.choose_imit(stu_rewards, learner.lr_, hard = True)

        if mode == 'omni' or random_prob is not None:
            w = learner.learn(teacher.mini_batch_indices_, teacher.mini_batch_opt_acts_, data_idx,
                                         gradients, i, teacher.stu_gt_reward_param_, random_prob)
        elif mode == 'omni_cont':
            w = learner.learn_cont(teacher.mini_batch_indices_, teacher.mini_batch_opt_acts_, data_idx,
                                         gradients, i, teacher.stu_gt_reward_param_, learner.config_.cont_K)
        elif mode == 'imit':
            w = learner.learn_imit(teacher.mini_batch_indices_, teacher.mini_batch_opt_acts_, data_idx,
                                              l_stu, i, teacher.stu_gt_reward_param_)
        elif mode == 'imit_cont':
            w = learner.learn_imit_cont(teacher.mini_batch_indices_, teacher.mini_batch_opt_acts_, data_idx,
                                                   l_stu, i, teacher.stu_gt_reward_param_)
        dists_.append(np.sqrt(np.sum(np.square(learner.current_mean_ - teacher.stu_gt_reward_param_))))
        dists.append(np.mean(np.max(abs(learner.current_action_prob() - teacher.action_probs_), axis = 1)))
        distsq.append(np.mean(np.square(learner.q_map_ - teacher.q_map_)))        
        ws.append(copy.deepcopy(w))
        #print(learner.q_map_)
        policy.append(copy.deepcopy(learner.q_map_))
        if (i + 1) % 2 == 0:
            actual_rewards.append(teacher.map_.test_walk(teacher.reward_param_, learner.action_probs_, test_set[i + 1], greedy = True))
    learner.lr_ = learner.config_.lr
    if (mode == "omni_cont"):
        np.save('action_probs.npy', learner.action_probs_)

    #np.save("learned_policy%d.npy" % (int(sys.argv[2])), np.asarray(policy))
    #np.save("learned_rewards%d.npy" % (int(sys.argv[2])), np.asarray(learned_rewards))
    #np.save('batch%d.npy' % (int(sys.argv[2])), batches)
    return dists, dists_, distsq, actual_rewards, ws, teaching_examples

def main():    
    exec('import config', globals())
    exec('from config import config_T', globals())
    exec('from config import config_L', globals())
    
    use_tf = config.use_tf
    mode = config.mode

    train_iter = config.train_iter

    #seed = int(sys.argv[2])
    seed = 1
    np.random.seed((seed + 1) * 159)

    np.set_printoptions(precision = 4)

    #map_num = int(sys.argv[3])
    map_num = 6
    feedback = False
    blockPrint()
    if (feedback):
        print(feedback)
    if not use_tf:
        np.random.seed((seed + 1) * 157)
       
        import tensorflow as tf       
        tfconfig = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)
        tfconfig.gpu_options.allow_growth = True
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        sess = tf.Session(config = tfconfig)

        map_l = Map(sess, config_L)
        np.random.seed((seed + 1) * 163)

        map_t = Map(sess, config_T)
        np.random.seed((seed + 1) * 174)
        enablePrint()         
        gt_r_param_tea = map_l.reward_grid(map_num) #map_l.reward_generate(3) if config.reward_type == 'E' else np.random.uniform(-2, 2, size = [1, config.shape ** 2])
        gt_r_param_stu = copy.deepcopy(gt_r_param_tea)
        if config_L.shuffle_state_feat:
            #print("Shuffling with ", map_l.feat_idx_)
            gt_r_param_stu[:, map_l.feat_idx_] = gt_r_param_tea

        assert(np.max(abs(np.sum(gt_r_param_stu * map_l.state_feats_, axis = 1) - np.sum(gt_r_param_tea * map_t.state_feats_, axis = 1))) < 1e-9)

        np.random.seed((seed + 1) * 105)
        teacher = TeacherIRL(sess, map_t, config_T, gt_r_param_tea, gt_r_param_stu)
        init_ws = np.random.uniform(-2, 2, size = [config_L.particle_num, teacher.map_.num_states_])
        unshuffled_ws = copy.deepcopy(init_ws)

        test_set = np.random.choice(teacher.map_.num_states_, size = [train_iter + 1, teacher.map_.num_states_ * 20])

    enablePrint()
    print('Wait for teacher test walk...')
    blockPrint()
    if mode == 'omni':
        teacher_rewards = []
        for i in range(0, train_iter, 2):
            teacher_rewards.append(teacher.map_.test_walk(teacher.reward_param_, teacher.action_probs_, test_set[i + 1], greedy = True))
        teacher_reward = np.asarray([np.mean(teacher_rewards)])
    

    learner = LearnerIRL(sess, map_l, config_L)

    human = learn(teacher, learner, '%s_cont' % mode, init_ws, train_iter, test_set, gt_r_param_tea, None, True, feedback)
    prag_cont = learn(teacher, learner, '%s_cont' % mode, init_ws, train_iter, test_set, gt_r_param_tea, None, False)
    imt = learn(teacher, learner, '%s_cont' % mode, init_ws, train_iter, test_set, gt_r_param_tea, 1, False)

    if False: 
        # do imt_human
        imt_human = learn(teacher, learner, '%s_cont' % mode, init_ws, train_iter, test_set, gt_r_param_tea, 1, True)


    fig, axs = plt.subplots(2,2, figsize=(10,10), constrained_layout=True)
    axs[0, 0].plot(prag_cont[0])
    axs[0, 0].plot(human[0])
    axs[0, 0].plot(imt[0])
    # axs[0, 0].plot(imt_human[0])

    axs[0, 0].set_title('action prob total variation distance')

    axs[0, 1].plot(prag_cont[1], label='ITAL')
    axs[0, 1].plot(human[1], label='ITAL, Human')
    axs[0, 1].plot(imt[1], label='IMT')
    # axs[0, 1].plot(imt_human[1], label='IMT, Human')

    axs[0, 1].legend()

    axs[0, 1].set_title('reward param l2 dist')

    axs[1, 0].set_title('q l2 dist')
    axs[1, 0].plot(prag_cont[2])
    axs[1, 0].plot(human[2])
    axs[1, 0].plot(imt[2])
    # axs[1, 0].plot(imt_human[2])

    axs[1, 1].plot(prag_cont[3])
    axs[1, 1].plot(human[3])
    axs[1, 1].plot(imt[3])
    # axs[1, 1].plot(imt_human[3])

    axs[1, 1].set_title('actual rewards (every 2 iter)')
    axs[1, 1].plot([teacher_reward] * len(prag_cont[3]))
    fig.suptitle('Discrete Rewards w/ Random Seed %d' % (seed))

    plt.savefig('fig%d_%d.png' % (seed, map_num))    
    plt.show()
    np.save('teaching_examples%d.npy' % (seed), np.asarray(prag_cont[-1])) 
    results = [prag_cont, imt, prag_cont]
    n = 3
    for i in range(n):
        dists = results[i][0]
        dists_ = results[i][1]
        distsq = results[i][2]
        ar = results[i][3]
        mat = results[i][4]
        np.save('Experiments/' + directory + "action_dist%d_%d" % (i, seed), dists, allow_pickle=True)
        np.save('Experiments/' + directory + "reward_dist%d_%d" % (i, seed), np.sqrt(dists_), allow_pickle=True)
        np.save('Experiments/' + directory + "q_dist%d_%d" % (i, seed), distsq, allow_pickle=True)
        np.save('Experiments/' + directory + "rewards%d_%d" % (i, seed), ar, allow_pickle=True)
        np.save('Experiments/' + directory + "matrix%d_%d" % (i, seed), mat, allow_pickle = True)

if __name__ == '__main__':
    main()
