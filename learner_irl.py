import numpy as np
import copy

class LearnerIRL:
    def __init__(self, map_input, config):
        self.config_ = config
        self.lr_ = self.config_['lr']
        self.map_ = map_input
        self.particles_ = np.random.uniform(-2, 2, size = [1, 5 ** 2])
        self.current_mean_ = np.mean(self.particles_, 0, keepdims = True)
        self.initial_val_maps_ = {}
        self.initial_valg_maps_ = {}
        self.value_iter_op_ = self.map_.value_iter
        self.gradient_iter_op_ = self.map_.grads_iter
        for i in range(self.config_['particle_num'] + 1):
            self.initial_val_maps_[i] = np.random.uniform(0, 1, size = [self.map_.num_states_, 1])
            self.initial_valg_maps_[i] = np.random.uniform(-1, 1, size = [self.map_.num_states_, self.map_.num_states_])

    def reset(self, init_ws):
        self.particles_ = copy.deepcopy(init_ws)
        self.current_mean_ = np.mean(self.particles_, 0, keepdims = True)
        for i in range(self.config_['particle_num'] + 1):
            self.initial_val_maps_[i] = np.random.uniform(0, 1, size = [self.map_.num_states_, 1])
            self.initial_valg_maps_[i] = np.random.uniform(-1, 1, size = [self.map_.num_states_, self.map_.num_states_])
    
    def learn_cont(self, mini_batch_indices, opt_actions, data_idx, gradients, step, gt_w, K = None, batch = False):
        prev_mean = copy.deepcopy(self.current_mean_)
        exp_cache_prev_func = lambda w_est: -1 * self.config_['beta_select'] *\
                                            ((self.config_['lr'] ** 2) * np.sum(np.square(gradients), axis = 1) +\
                                            2 * self.config_['lr'] * np.sum((prev_mean - w_est) * gradients, axis = 1))
        exp_cache_func = lambda vals: np.exp(vals - np.max(vals))
        teacher_sample_lle_func = lambda exps: np.log((exps / np.sum(exps))[data_idx])
        lle_gradient_func = lambda exps: 2 * self.config_['beta_select'] * self.config_['lr'] * gradients[data_idx: data_idx + 1, ...] -\
                                                2 * self.config_['beta_select'] * self.config_['lr'] * np.sum(gradients *\
                                                np.expand_dims(exps, -1) / np.sum(exps), axis = 0, keepdims = True)
        
        def get_grad():
            val_map, q_map, _ = self.value_iter_op_(self.current_mean_, value_map_init = self.initial_val_maps_[0], hard_max = True)
            if np.sum(np.isnan(val_map)) > 0:
                pdb.set_trace()
            self.initial_val_maps_[0] = val_map
            valg_map, qg_map, _ = self.gradient_iter_op_(q_map, value_map_init = self.initial_valg_maps_[0])
            if np.sum(np.isnan(valg_map)) > 0:
                print("STEP", step)
                print("PARTICLE NUM", i)
                pdb.set_trace()
            self.initial_valg_maps_[0] = valg_map
            
            if batch:
                exp_q = np.exp(self.config_['beta'] * q_map[mini_batch_indices, ...])
                action_prob = exp_q / np.sum(exp_q, axis = 1, keepdims = True)
                gradients = self.config_['beta'] * (qg_map[mini_batch_indices, opt_actions, ...] -\
                                                 np.sum(np.expand_dims(action_prob, 2) * qg_map[mini_batch_indices, ...], axis = 1))
                return np.mean(gradients, axis = 0, keepdims = True)

            action_q = q_map[mini_batch_indices[data_idx]: mini_batch_indices[data_idx] + 1, ...]
            exp_q = np.exp(self.config_['beta'] * (action_q - np.max(action_q)))
            action_prob = exp_q / np.sum(exp_q, axis = 1, keepdims = True)
            particle_gradient = self.config_['beta'] * (qg_map[mini_batch_indices[data_idx], opt_actions[data_idx]: opt_actions[data_idx] + 1, ...] -\
                                                        np.sum(np.expand_dims(action_prob, 2) * qg_map[mini_batch_indices[data_idx]: mini_batch_indices[data_idx] + 1, ...], axis = 1))
            if np.sum(np.isnan(particle_gradient)) > 0:
                pdb.set_trace()
            if np.sum(particle_gradient != 0) == 0:
                pdb.set_trace()
            return particle_gradient
        
        def get_new_lle():
            val_map, q_map, _ = self.value_iter_op_(self.current_mean_, value_map_init = self.initial_val_maps_[0], hard_max = True)
            self.initial_val_maps_[0] = val_map
            plle = self.config_['beta'] * q_map[(mini_batch_indices, opt_actions)] -\
                   np.log(np.sum(np.exp(self.config_['beta'] * q_map[mini_batch_indices, ...]), axis = 1))
            return plle

        if K > 0:
            for i in range(K):
                gradient_tf = get_grad()
                self.current_mean_ += self.config_['lr'] * gradient_tf
                lle_gradient = lle_gradient_func(exp_cache_func(exp_cache_prev_func(self.current_mean_)))
                self.current_mean_ += self.config_['lr'] * lle_gradient
        else:
            for i in range(abs(K)):
                gradient_tf = get_grad()
                self.current_mean_ += self.config_['lr'] * gradient_tf

        return self.current_mean_

    def current_action_prob(self):
        _, self.q_map_, _ = self.value_iter_op_(self.current_mean_, hard_max = True)
                                          #value_map_init = self.initial_val_maps_[self.config_.particle_num])
        q_balance = self.q_map_ - np.max(self.q_map_, axis = 1, keepdims = True)
        exp_q = np.exp(self.config_['beta'] * q_balance)
        self.action_probs_ = exp_q / np.sum(exp_q, axis = 1, keepdims = True)
        return self.action_probs_
