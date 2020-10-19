import numpy as np
from Game.map import Map

import pdb

class TeacherIRL:
    def __init__(self, map_input, config, gt_r_param, stu_gt_r_param):
        self.config_ = config
        self.reward_param_ = gt_r_param
        self.stu_gt_reward_param_ = stu_gt_r_param
        self.map_ = map_input
        self.value_iter_op_ = self.map_.value_iter
        self.gradient_iter_op_ = self.map_.grads_iter
        self.rewards_ = np.sum(self.map_.state_feats_ * self.reward_param_, axis = 1)
        self.value_map_, self.q_map_, _ = self.value_iter_op_(self.reward_param_)
        value_map_max, _, _ = self.value_iter_op_(self.reward_param_, hard_max = True)
        #the level of approximation k might need to be tuned given different rewards scale
        assert(np.max(abs(value_map_max - self.value_map_)) / np.mean(abs(value_map_max)) < 0.02)
        q_balance = self.q_map_ - np.mean(self.q_map_, axis = 1, keepdims = True)
        exp_q = np.exp(self.config_['beta'] * q_balance)
        self.action_probs_ = exp_q / np.sum(exp_q, axis = 1, keepdims = True)
        self.mini_batch_indices_ = None
        self.l_ = self.config_['beta'] * self.q_map_ - np.log(np.sum(np.exp(self.config_['beta'] * self.q_map_), axis = 1, keepdims = True))
        self.initial_val_maps_ = np.random.uniform(0, 1, size = [self.map_.num_states_, 1])
        self.initial_valg_maps_ = np.random.uniform(-1, 1, size = [self.map_.num_states_, self.map_.num_states_])

        self.mini_batch_indices_ = [] 

    def sample(self):
        previous = self.mini_batch_indices_
        possible_indices = list(set(range(self.map_.num_states_)).difference(set(previous)))

        self.mini_batch_indices_ = np.random.choice(possible_indices, size = self.config_['sample_size'], replace=False)

        self.mini_batch_opt_acts_ = []
        for idx in self.mini_batch_indices_:
            self.mini_batch_opt_acts_.append(np.random.choice(len(self.map_.actions_), p = self.action_probs_[idx, ...]))
        self.mini_batch_opt_acts_ = np.array(self.mini_batch_opt_acts_)

        return
    
    def choose(self, learner_param, lr, hard = True):
        assert(self.mini_batch_indices_ is not None)
        val_map, q_map, _ = self.value_iter_op_(learner_param, value_map_init = self.initial_val_maps_)
        self.initial_val_maps_ = val_map
        valg_map, qg_map, _ = self.gradient_iter_op_(q_map, value_map_init = self.initial_valg_maps_)
        self.initial_valg_maps_  = valg_map
        
        exp_q = np.exp(self.config_['beta'] * q_map[self.mini_batch_indices_, ...])
        action_prob = exp_q / np.sum(exp_q, axis = 1, keepdims = True)
        gradients = self.config_['beta'] * (qg_map[self.mini_batch_indices_, self.mini_batch_opt_acts_, ...] -\
                                         np.sum(np.expand_dims(action_prob, 2) * qg_map[self.mini_batch_indices_, ...], axis = 1))
        
        vals = -1 * self.config_['beta_select'] * (np.sum(lr * lr * np.square(gradients), axis = 1) + 2 * lr * np.sum((learner_param - self.reward_param_) * gradients, axis = 1))
        if hard: 
            return np.argmax(vals), gradients
        vals -= np.max(vals)
        logits = np.exp(vals)
        if np.sum(np.isnan(logits)) > 0:
            pdb.set_trace()
        selected = np.random.choice(len(vals), 1, p = logits / np.sum(logits))[0]
        # return np.argmin(vals)
        return selected, gradients