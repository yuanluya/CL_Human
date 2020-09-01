import numpy as np
from game import Game
import copy

class LearnHuman:
    def __init__(self, teacher, learner, init_ws, test_set, teacher_rewards):
        self.teacher = teacher
        self.learner = learner
        self.init_ws = init_ws
        self.mode = '%omni_cont'
        self.test_set = test_set
        self.teacher_rewards = teacher_rewards
        self.feedback = True
        
        self.iteration_limit = 30
        self.step = 0
        self.batches = []
        self.ws = []
        self.learned_rewards = []
        self.policy = []
        self.selected_indices = []
        
    def chooseIdx(self):                
        self.step += 1
        self.teacher.sample()
        self.learned_rewards.append(copy.deepcopy(self.learner.current_mean_))
        self.batches.append([self.teacher.mini_batch_indices_, self.teacher.mini_batch_opt_acts_])
        action_probs = self.learner.current_action_prob()
        data_idx, self.gradients = self.teacher.choose(self.learner.current_mean_, self.learner.lr_, hard = True)
        
        if self.feedback:
            self.g = Game(self.teacher_rewards, self.batches[-1], copy.deepcopy(self.learner.q_map_), copy.deepcopy(self.learner.current_mean_), self.step, data_idx)
        else:
            self.g = Game(teacher_rewards, batches[-1], copy.deepcopy(learner.q_map_), copy.deepcopy(learner.current_mean_))

        self.g.display()

    def updateLearner(self):
        if self.g.selected_idx_==None:
            self.reset()
            print('Please select an arrow before running a new iteration.')
            exit()
        data_idx = self.g.selected_idx_
        self.selected_indices.append(data_idx)
        w = self.learner.learn_cont(self.teacher.mini_batch_indices_, self.teacher.mini_batch_opt_acts_, data_idx,
                                         self.gradients, self.step, self.teacher.stu_gt_reward_param_, self.learner.config_['cont_K'])        
              
        self.ws.append(copy.deepcopy(w))
        #print(learner.q_map_)
        self.policy.append(copy.deepcopy(self.learner.q_map_))
        
    def saveData(self):
        data = {
                'batches': self.batches,
                'ws': self.ws,
                'learned_rewards': self.learned_rewards,
                'policy': self.policy,
                'selected_indices': self.selected_indices
               }
        np.save('data.npy', data, allow_pickle=True)
      
    def reset(self):
        self.step = 0
        self.batches = []
        self.ws = []
        self.learned_rewards = []
        self.policy = []
        self.selected_indices = []
        
    def iteration(self):
        if self.step > 0:
            self.updateLearner()
        if self.step < self.iteration_limit:
            self.chooseIdx()
    