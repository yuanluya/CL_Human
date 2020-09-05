import numpy as np
from game import Game
import copy
from session import Session

class LearnHuman:
    def __init__(self, teacher, learner, init_ws, test_set, teacher_rewards, train_iter, feedback, map_num, s, random_prob = None):
        self.teacher = teacher
        self.learner = learner
        self.learner.reset(init_ws)
        self.init_ws = init_ws
        self.mode = '%omni_cont'
        self.test_set = test_set
        self.teacher_rewards = teacher_rewards
        self.feedback = feedback
        self.map_num = map_num
        self.sess = s
        self.random_prob = random_prob

        self.iteration_limit = train_iter
        self.idx_selected = True
        self.step = 0
        self.batches = []
        self.ws = [copy.deepcopy(init_ws)]
        self.learned_rewards = []
        self.policy = []
        self.selected_indices = []
        
    def chooseIdx(self):                
        if self.idx_selected:
            self.step += 1
            self.teacher.sample()
            self.learned_rewards.append(copy.deepcopy(self.learner.current_mean_))
            self.batches.append([self.teacher.mini_batch_indices_, self.teacher.mini_batch_opt_acts_])
            action_probs = self.learner.current_action_prob()
        else:
            self.idx_selected = True
        data_idx, self.gradients = self.teacher.choose(self.learner.current_mean_, self.learner.lr_, hard = True)
        
        if self.feedback:
            self.g = Game(self.teacher_rewards, self.batches[-1], copy.deepcopy(self.learner.q_map_),
                          copy.deepcopy(self.learner.current_mean_), self.step, self.iteration_limit, data_idx)
        else:
            self.g = Game(self.teacher_rewards, self.batches[-1], copy.deepcopy(self.learner.q_map_),
                          copy.deepcopy(self.learner.current_mean_), self.step, self.iteration_limit)

        self.g.display()

    def updateLearner(self):
        if self.g.selected_idx_==None:
            self.idx_selected = False
            print('Please select an arrow before running a new iteration.')
            return
        data_idx = self.g.selected_idx_
        self.selected_indices.append(data_idx)
        if self.random_prob == None:
            w = self.learner.learn_cont(self.teacher.mini_batch_indices_, self.teacher.mini_batch_opt_acts_, data_idx,
                                         self.gradients, self.step, self.teacher.stu_gt_reward_param_, self.learner.config_['cont_K'])        
        else:
            w = self.learner.learn(self.teacher.mini_batch_indices_, self.teacher.mini_batch_opt_acts_, data_idx,
                                         self.gradients, self.step, self.teacher.stu_gt_reward_param_, self.random_prob)          
        self.ws.append(copy.deepcopy(w))
        #print(learner.q_map_)
        self.policy.append(copy.deepcopy(self.learner.q_map_))
        if self.step == self.iteration_limit:
            print('All iterations are completed.')
    def saveData(self):
        data = {
                'batches': self.batches,
                'ws': self.ws,
                'learned_rewards': self.learned_rewards,
                'policy': self.policy,
                'selected_indices': self.selected_indices
               }
        np.save('data/data%d.npy' % (self.sess.random_seed), data, allow_pickle=True)

        self.sess.save_data() 
        
    def reset(self):
        self.step = 0
        self.learner.reset(self.init_ws)
        self.batches = []
        self.ws = []
        self.learned_rewards = []
        self.policy = []
        self.selected_indices = []
        
    def iteration(self):
        if (self.step > 0) and (self.step <= self.iteration_limit):
            self.updateLearner()
        if self.step < self.iteration_limit:
            self.chooseIdx()
    