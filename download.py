from __future__ import print_function
import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from apiclient.http import MediaFileUpload
from apiclient.http import MediaIoBaseUpload
from apiclient.http import MediaIoBaseDownload
from apiclient import errors 
import io
import numpy as np
from map import Map
from teacher_irl import TeacherIRL
from learner_irl import LearnerIRL
from multiprocessing import Process, Manager
import copy
import sys
from easydict import EasyDict as edict
from tqdm import tqdm
import matplotlib.pyplot as plt
if plt.get_backend() == 'Qt5Agg':
    from matplotlib.backends.qt_compat import QtWidgets
    qApp = QtWidgets.QApplication(sys.argv)
    plt.matplotlib.rcParams['figure.dpi'] = qApp.desktop().physicalDpiX()
import time
import pdb

from game import Game

import seaborn as sns
import pandas as pd
from collections import OrderedDict
import matplotlib.colors as mcolors
import scipy.stats
class DataDownload:
    def __init__(self, seed, map_num):
        self.seed = seed
        self.map_num = map_num
        self.download() 

    def download(self):
        seed = self.seed
        map_num = self.map_num

        SCOPES = ['https://www.googleapis.com/auth/drive']
        creds = None
        # The file token.pickle stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)

        service = build('drive', 'v3', credentials=creds)
        print("Service built.")

        page_token = None
        while True:
            response = service.files().list(q= "mimeType = 'application/vnd.google-apps.folder' and name = 'Map %d'" % (map_num),
                                                  spaces='drive',
                                                  fields='nextPageToken, files(id, name)',
                                                  pageToken=page_token).execute()
            for file in response.get('files', []):
                print('Found file: %s (%s)' % (file.get('name'), file.get('id')))
                folder_id = file.get('id')

            page_token = response.get('nextPageToken', None)
            if page_token is None:
                break

        page_token = None
        while True:
            response = service.files().list(q="name = 'data%d_ital.npy' and '%s' in parents" % (seed, folder_id),
                                                  spaces='drive',
                                                  fields='nextPageToken, files(id, name)',
                                                  pageToken=page_token).execute()
            for file in response.get('files', []):
                print('Found file: %s (%s)' % (file.get('name'), file.get('id')))
                file_id = file.get('id')

            page_token = response.get('nextPageToken', None)
            if page_token is None:
                break

        page_token = None
        while True:
            response = service.files().list(q="name = 'data%d_imt.npy' and '%s' in parents" % (seed, folder_id),
                                                  spaces='drive',
                                                  fields='nextPageToken, files(id, name)',
                                                  pageToken=page_token).execute()
            for file in response.get('files', []):
                print('Found file: %s (%s)' % (file.get('name'), file.get('id')))
                imt_id = file.get('id')

            page_token = response.get('nextPageToken', None)
            if page_token is None:
                break

        request = service.files().get_media(fileId=file_id)

        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()

        with open("map_%d_data%d_ital.npy" % (map_num, seed), "wb") as f:
            f.write(fh.getbuffer())

        request2 = service.files().get_media(fileId=imt_id)
        fh2 = io.BytesIO()
        downloader2 = MediaIoBaseDownload(fh2, request2)
        done = False
        while done is False:
            status, done = downloader2.next_chunk()

        with open("map_%d_data%d_imt.npy" % (map_num, seed), "wb") as f:
            f.write(fh2.getbuffer())

    def learn(self, teacher, learner, mode, init_ws, train_iter, test_set, teacher_rewards, data, random_prob = None, human = False):

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

        for i in tqdm(range(train_iter)):
            teacher.mini_batch_indices_ = data['batches'][i][0]
            teacher.mini_batch_opt_acts_ = data['batches'][i][1]

            learned_rewards.append(copy.deepcopy(learner.current_mean_))
            if mode[0: 4] == 'omni':
                data_idx, gradients = teacher.choose(learner.current_mean_, learner.lr_, hard = True)
                # teaching_examples.append([teacher.mini_batch_indices_[data_idx], teacher.mini_batch_opt_acts_[data_idx]])
                if (human):
                    data_idx = data['selected_indices'][i]

            elif mode[0: 4] == 'imit':
                stu_rewards = np.sum(learner.map_.state_feats_ * learner.current_mean_, axis = 1, keepdims = True)
                data_idx, gradients, l_stu = teacher.choose_imit(stu_rewards, learner.lr_, hard = True)

            if mode == 'omni' or random_prob is not None:
                w = learner.learn(teacher.mini_batch_indices_, teacher.mini_batch_opt_acts_, data_idx,
                                             gradients, i, teacher.stu_gt_reward_param_, random_prob)
            elif mode == 'omni_cont':
                w = learner.learn_cont(teacher.mini_batch_indices_, teacher.mini_batch_opt_acts_, data_idx,
                                             gradients, i, teacher.stu_gt_reward_param_, learner.config_['cont_K'])
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

        if (mode == "omni_cont"):
            np.save('action_probs.npy', learner.action_probs_)

        #np.save("learned_policy%d.npy" % (int(sys.argv[2])), np.asarray(policy))
        #np.save("learned_rewards%d.npy" % (int(sys.argv[2])), np.asarray(learned_rewards))
        #np.save('batch%d.npy' % (int(sys.argv[2])), batches)
        return dists, dists_, distsq, actual_rewards, ws, teaching_examples

    def graph_data(self):

        data_ital = np.load("map_%d_data%d_ital.npy" % (self.map_num, self.seed), allow_pickle = True)[()]
        data_imt = np.load("map_%d_data%d_imt.npy"  % (self.map_num, self.seed), allow_pickle = True)[()]

        use_tf = False
        multi_thread = False
        feedback = True

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

        config_T = {'shape': shape, 'approx_type': 'gsm', 'beta': beta, 'shuffle_state_feat': False,
                          'lr': lr, 'sample_size': 10, 'use_tf': use_tf, 'approx_k': approx_k, 'beta_select': beta_select}
        config_L = {'shape': shape, 'approx_type': 'gsm', 'beta': beta, 'lr': lr, "prob": 1,
                          'shuffle_state_feat': mode == 'imit', 'particle_num': 1, 'replace_count': 1,
                          'noise_scale_min': noise_scale_min, 'noise_scale_max': noise_scale_max, 'noise_scale_decay': noise_scale_decay, 'cont_K': K,
                          'target_ratio': 0, 'new_ratio': 1, 'use_tf': use_tf, 'approx_k': approx_k, 'beta_select': beta_select}


        np.set_printoptions(precision = 4)

        map_l = Map(config_L)
        map_t = Map(config_T)

        gt_r_param_tea = map_l.reward_grid(self.map_num)
        gt_r_param_stu = copy.deepcopy(gt_r_param_tea)

        test_set = np.random.choice(25, size = [30 + 1, 25 * 20])
        init_ws = data_ital['ws'][0]
        

        teacher = TeacherIRL(map_t, config_T, gt_r_param_tea, gt_r_param_stu)

        if mode == 'omni':
            teacher_rewards = []
            for i in tqdm(range(0, train_iter, 2)):
                teacher_rewards.append(teacher.map_.test_walk(teacher.reward_param_, teacher.action_probs_, test_set[i + 1], greedy = True))
            teacher_reward = np.asarray([np.mean(teacher_rewards)])
            # np.save('Experiments/' + directory + "teacher_rewards_%d" % (seed), teacher_rewards, allow_pickle=True)
        

        learner = LearnerIRL(map_l, config_L)

        human = self.learn(teacher, learner, '%s_cont' % mode, init_ws, train_iter, test_set, gt_r_param_tea, data_ital,  None, True)
        imt_human = self.learn(teacher, learner, '%s_cont' % mode, init_ws, train_iter, test_set, gt_r_param_tea, data_imt, 1, True)

        prag_cont = self.learn(teacher, learner, '%s_cont' % mode, init_ws, train_iter, test_set, gt_r_param_tea, data_ital, None)
        imt = self.learn(teacher, learner, '%s_cont' % mode, init_ws, train_iter, test_set, gt_r_param_tea, data_imt, 1)


        fig, axs = plt.subplots(2,2, figsize=(10,10), constrained_layout=True)
        axs[0, 0].plot(prag_cont[0])
        axs[0, 0].plot(human[0])
        axs[0, 0].plot(imt[0])
        axs[0, 0].plot(imt_human[0])

        axs[0, 0].set_title('action prob total variation distance')

        axs[0, 1].plot(prag_cont[1], label='ITAL')
        axs[0, 1].plot(human[1], label='ITAL, Human')
        axs[0, 1].plot(imt[1], label='IMT')
        axs[0, 1].plot(imt_human[1], label='IMT, Human')

        axs[0, 1].legend()

        axs[0, 1].set_title('reward param l2 dist')

        axs[1, 0].set_title('q l2 dist')
        axs[1, 0].plot(prag_cont[2])
        axs[1, 0].plot(human[2])
        axs[1, 0].plot(imt[2])
        axs[1, 0].plot(imt_human[2])

        axs[1, 1].plot(prag_cont[3])
        axs[1, 1].plot(human[3])
        axs[1, 1].plot(imt[3])
        axs[1, 1].plot(imt_human[3])

        axs[1, 1].set_title('actual rewards (every 2 iter)')
        axs[1, 1].plot([teacher_reward] * len(prag_cont[3]))

        plt.suptitle("Seed %d, Map %d" % (self.seed, self.map_num))
        plt.savefig('figure%d_%d.png' % (self.seed, self.map_num))    
        plt.show()


def learn(teacher, learner, mode, init_ws, train_iter, test_set, teacher_rewards, data, random_prob = None, human = False, rand_tea = False):
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

    for i in tqdm(range(train_iter)):
        teacher.mini_batch_indices_ = data['batches'][i][0]
        teacher.mini_batch_opt_acts_ = data['batches'][i][1]

        learned_rewards.append(copy.deepcopy(learner.current_mean_))

        data_idx, gradients = teacher.choose(learner.current_mean_, learner.lr_, hard = True)
        if (human):
            data_idx = data['selected_indices'][i]
        elif rand_tea:
            data_idx = np.random.randint(teacher.config_['sample_size'])
        if mode == 'omni' or random_prob is not None:
            w = learner.learn(teacher.mini_batch_indices_, teacher.mini_batch_opt_acts_, data_idx,
                                         gradients, i, teacher.stu_gt_reward_param_, random_prob)
        elif mode == 'omni_cont':
            w = learner.learn_cont(teacher.mini_batch_indices_, teacher.mini_batch_opt_acts_, data_idx,
                                         gradients, i, teacher.stu_gt_reward_param_, learner.config_['cont_K'])

        dists_.append(np.sqrt(np.sum(np.square(learner.current_mean_ - teacher.stu_gt_reward_param_))))
        dists.append(np.mean(np.max(abs(learner.current_action_prob() - teacher.action_probs_), axis = 1)))
        distsq.append(np.mean(np.square(learner.q_map_ - teacher.q_map_)))        
        ws.append(copy.deepcopy(w))
        #print(learner.q_map_)
        policy.append(copy.deepcopy(learner.q_map_))

        if (i + 1) % 2 == 0:
            actual_rewards.append(teacher.map_.test_walk(teacher.reward_param_, learner.action_probs_, test_set[i + 1], greedy = True))

    return np.array(dists), np.array(dists_), np.array(distsq), np.array(actual_rewards), ws, teaching_examples

def download_all(seed_l, map_num):
    
    downloaded = []
    for s in seed_l:
        try:
            DataDownload(s, map_num)
            downloaded.append(s)
        except UnboundLocalError:
            continue
    return downloaded

def CollectData(map_num):
    map_download = {0:[1, 7, 8, 10, 11, 12, 21, 22, 26, 27, 29, 39, 48, 52, 56, 57, 58, 60, 61, 62, 63],
                    1: [4, 6, 8, 10, 11, 12, 21, 22, 26, 27, 29, 40, 51, 56, 57, 58, 60, 61, 62, 63] ,
                    3: [1, 4, 8, 13, 14, 21, 26, 27, 30, 34, 41, 51, 53, 56, 57, 58, 60, 61, 63, 64] ,
                    4: [1, 4, 6, 8, 13, 14, 21, 26, 27, 34, 43, 51, 53, 56, 57, 58, 60, 61, 64, 65] ,
                    7: [1, 4, 6, 8, 13, 14, 21, 26, 27, 34, 43, 49, 51, 53, 56, 57, 58, 60, 61, 64] 
                    }
    downloaded = download_all(map_download[map_num], map_num)
    assert(downloaded == map_download[map_num])
    if not downloaded:
        print("No seeds downloaded.")
        return
    exec('import config', globals())
    exec('from config import config_T', globals())
    exec('from config import config_L', globals())
    np.set_printoptions(precision = 4)

    map_l = Map(config_L)
    map_t = Map(config_T)

    gt_r_param_tea = map_l.reward_grid(map_num)
    gt_r_param_stu = copy.deepcopy(gt_r_param_tea)

    test_set = np.random.choice(25, size = [30 + 1, 25 * 20])

    teacher = TeacherIRL(map_t, config_T, gt_r_param_tea, gt_r_param_stu)
    
    teacher_rewards = []
    for i in tqdm(range(0, config.train_iter, 2)):
        teacher_rewards.append(teacher.map_.test_walk(teacher.reward_param_, teacher.action_probs_, test_set[i + 1], greedy = True))
    teacher_reward = np.asarray([np.mean(teacher_rewards)])
    np.save("data/map%d/teacher_reward.npy"% map_num, teacher_reward)
    
    learner = LearnerIRL(map_l, config_L)

    human_acc = [[],[],[],[]]
    imt_human_acc = [[],[],[],[]]
    prag_cont_acc = [[],[],[],[]]
    imt_acc = [[],[],[],[]]
    rand_ital = [[],[],[],[]]
    for s in downloaded[0:]:
        data_ital = np.load("map_%d_data%d_ital.npy" % (map_num, s), allow_pickle = True)[()]
        data_imt = np.load("map_%d_data%d_imt.npy"  % (map_num, s), allow_pickle = True)[()] 
        init_ws = data_ital['ws'][0]
        
        human = learn(teacher, learner, '%s_cont' % config.mode, init_ws, config.train_iter, test_set, gt_r_param_tea, data_ital,  None, True)
        imt_human = learn(teacher, learner, '%s_cont' % config.mode, init_ws, config.train_iter, test_set, gt_r_param_tea, data_imt, 1, True)
        prag_cont = learn(teacher, learner, '%s_cont' % config.mode, init_ws, config.train_iter, test_set, gt_r_param_tea, data_ital, None)
        imt = learn(teacher, learner, '%s_cont' % config.mode, init_ws, config.train_iter, test_set, gt_r_param_tea, data_imt, 1)    
        
        rand = learn(teacher, learner, '%s_cont' % config.mode, init_ws, config.train_iter, test_set, gt_r_param_tea, data_ital, None, rand_tea=True)
        for i in range(4):
            
            human_acc[i].append(human[i].flatten().astype(float))
            imt_human_acc[i].append(imt_human[i].flatten().astype(float))
            prag_cont_acc[i].append(prag_cont[i].flatten().astype(float))
            imt_acc[i].append(imt[i].flatten().astype(float))
            
            rand_ital[i].append(rand[i].flatten().astype(float))
    
    np.save("data/map%d/machine_ital.npy"% map_num, prag_cont_acc)
    np.save("data/map%d/machine_imt.npy"% map_num, imt_acc)
    np.save("data/map%d/human_ital.npy"% map_num, human_acc)
    np.save("data/map%d/human_imt.npy"% map_num, imt_human_acc)
    
    np.save("data/map%d/rand_ital.npy"% map_num, rand_ital)
def plot_average_SC(map_num, category_idx):
    confidence = 0.95
    linewidth = 2
    plt.rc('xtick', labelsize=20) 
    plt.rc('ytick', labelsize=20) 

    prag_cont_array = np.load("data/map%d/machine_ital.npy"% map_num, allow_pickle = True)[()]
    imt_array = np.load("data/map%d/machine_imt.npy"% map_num, allow_pickle = True)[()]
    human_array = np.load("data/map%d/human_ital.npy"% map_num, allow_pickle = True)[()]
    imt_human_array = np.load("data/map%d/human_imt.npy"% map_num, allow_pickle = True)[()]
    rand_ital_array = np.load("data/map%d/rand_ital.npy"% map_num, allow_pickle = True)[()]
    teacher_reward = np.load("data/map%d/teacher_reward.npy"% map_num, allow_pickle = True)[()]
    mapTochar = {0: 'A', 1: 'B', 3: 'C', 4: 'D', 7: 'E'}
    axis_title = {0: 'Total Policy Variance', 1: 'L2 Distance', 3: 'Actual Reward'}
    fig_title = {0: 'totalVariance', 1: 'l2', 3: 'rewards'}
    plt.figure()
    fig1, axs1 = plt.subplots(1,1, figsize=(10,6), constrained_layout=True)
    if category_idx == 3:
        axs1.plot([teacher_reward] * prag_cont_array[category_idx][0].shape[0], linewidth=linewidth, color='tab:grey')
    axs1.plot(np.mean(prag_cont_array[category_idx], axis = 0), linewidth=linewidth, color = 'tab:red')
    axs1.plot(np.mean(human_array[category_idx], axis = 0), linewidth=linewidth, color = 'tab:orange')
    axs1.plot(np.mean(imt_array[category_idx], axis = 0), linewidth=linewidth, color = 'tab:blue')
    axs1.plot(np.mean(imt_human_array[category_idx], axis = 0), linewidth=linewidth, color = 'tab:green')
    axs1.plot(np.mean(rand_ital_array[category_idx], axis = 0), linewidth=linewidth, color = 'tab:purple')
    axs1.set_title(axis_title[category_idx], fontweight="bold", size=29)
    x_range = 31 if category_idx-1 != 2 else 16

    m, l, h = mean_confidence_interval(prag_cont_array[category_idx], confidence=confidence)
    axs1.fill_between(np.arange(x_range), l, h,
                alpha=0.2, facecolor='tab:red', edgecolor=(0.8392156862745098, 0.15294117647058825, 0.1568627450980392,0.5)) 
    m, l, h = mean_confidence_interval(human_array[category_idx], confidence=confidence)
    axs1.fill_between(np.arange(x_range), l, h,
                 alpha=0.2, facecolor='tab:orange', edgecolor=(1.0, 0.4980392156862745, 0.054901960784313725,0.5)) 
    m, l, h = mean_confidence_interval(imt_array[category_idx], confidence=confidence)
    axs1.fill_between(np.arange(x_range), l, h,
                 alpha=0.2, facecolor='tab:blue', edgecolor=(0.12156862745098039, 0.4666666666666667, 0.7058823529411765,0.5))
    m, l, h = mean_confidence_interval(imt_human_array[category_idx], confidence=confidence)
    axs1.fill_between(np.arange(x_range), l, h,
                 alpha=0.2, facecolor='tab:green', edgecolor=(0.17254901960784313, 0.6274509803921569, 0.17254901960784313,0.5)) 
    m, l, h = mean_confidence_interval(rand_ital_array[category_idx], confidence=confidence)
    axs1.fill_between(np.arange(x_range), l, h,
                 alpha=0.2, facecolor='tab:brown', edgecolor=(0.5903921568627451, 0.403921568627451, 0.7411764705882353)) 
    axs1.set_xlabel('Training Iteration', fontweight="bold", size=29)
    plt.savefig('data/map%s_%s.pdf' % (mapTochar[map_num], fig_title[category_idx]), dpi=300) 


def plot_average(map_num):
    mapTochar = {0: 'A', 1: 'B', 3: 'C', 4: 'D', 7: 'E'}
    prag_cont_array = np.load("data/map%d/machine_ital.npy"% map_num, allow_pickle = True)[()]
    imt_array = np.load("data/map%d/machine_imt.npy"% map_num, allow_pickle = True)[()]
    human_array = np.load("data/map%d/human_ital.npy"% map_num, allow_pickle = True)[()]
    imt_human_array = np.load("data/map%d/human_imt.npy"% map_num, allow_pickle = True)[()]
    rand_ital_array = np.load("data/map%d/rand_ital.npy"% map_num, allow_pickle = True)[()]
    teacher_reward = np.load("data/map%d/teacher_reward.npy"% map_num, allow_pickle = True)[()]
    
    confidence = 0.95
    linewidth = 2
    plt.rc('xtick', labelsize=20) 
    plt.rc('ytick', labelsize=20) 
    fig, axs = plt.subplots(1,2, num='seaborn-darkgrid', figsize=(20,6), constrained_layout=True)
    axs[0].plot(np.mean(prag_cont_array[0], axis = 0), linewidth=linewidth, color = 'tab:red')
    axs[0].plot(np.mean(human_array[0], axis = 0), linewidth=linewidth, color = 'tab:orange')
    axs[0].plot(np.mean(imt_array[0], axis = 0), linewidth=linewidth, color = 'tab:blue')
    axs[0].plot(np.mean(imt_human_array[0], axis = 0), linewidth=linewidth, color = 'tab:green')
    axs[0].plot(np.mean(rand_ital_array[0], axis = 0), linewidth=linewidth, color = 'tab:purple')

    axs[0].set_title('Total Variation', fontweight="bold", size=29)
    axs[0].set_xlabel('Training Iteration', fontweight="bold", size=29)
    axs[1].set_xlabel('Training Iteration', fontweight="bold", size=29)
    axs[1].plot(np.mean(prag_cont_array[3], axis = 0), linewidth=linewidth, label='ITAL', color = 'tab:red')
    axs[1].plot(np.mean(human_array[3], axis = 0), linewidth=linewidth, label='ITAL, Human', color = 'tab:orange')
    axs[1].plot(np.mean(imt_array[3], axis = 0), linewidth=linewidth, label='IMT', color = 'tab:blue')
    axs[1].plot(np.mean(imt_human_array[3], axis = 0), linewidth=linewidth, label='IMT, Human', color = 'tab:green')
    axs[1].plot(np.mean(rand_ital_array[3], axis = 0), linewidth=linewidth, label='IMT, Rand', color = 'tab:purple')

    axs[1].set_title('Actual Rewards', fontweight="bold", size=29)
 
    for idx in [0, 3]:
        x_range = 31 if idx == 0 else 16
        m, l, h = mean_confidence_interval(prag_cont_array[idx], confidence=confidence)
        idx_ = idx if idx == 0 else 1
        axs[idx_].fill_between(np.arange(x_range), l, h,
                    alpha=0.2, facecolor='tab:red', edgecolor=(0.8392156862745098, 0.15294117647058825, 0.1568627450980392,0.5)) 
        m, l, h = mean_confidence_interval(human_array[idx], confidence=confidence)
        axs[idx_].fill_between(np.arange(x_range), l, h,
                     alpha=0.2, facecolor='tab:orange', edgecolor=(1.0, 0.4980392156862745, 0.054901960784313725,0.5)) 
        m, l, h = mean_confidence_interval(imt_array[idx], confidence=confidence)
        axs[idx_].fill_between(np.arange(x_range), l, h,
                     alpha=0.2, facecolor='tab:blue', edgecolor=(0.12156862745098039, 0.4666666666666667, 0.7058823529411765,0.5))
        m, l, h = mean_confidence_interval(imt_human_array[idx], confidence=confidence)
        axs[idx_].fill_between(np.arange(x_range), l, h,
                     alpha=0.2, facecolor='tab:green', edgecolor=(0.17254901960784313, 0.6274509803921569, 0.17254901960784313,0.5)) 
 
        m, l, h = mean_confidence_interval(rand_ital_array[idx], confidence=confidence)
        axs[idx_].fill_between(np.arange(x_range), l, h,
                     alpha=0.2, facecolor='tab:purple', edgecolor=(0.5903921568627451, 0.403921568627451, 0.7411764705882353))    
    axs[1].plot([teacher_reward] * prag_cont_array[3][0].shape[0], linewidth=linewidth, color='tab:grey')
    plt.savefig('data/map%d-supp.pdf' % (map_num), dpi=300) 

    plt.figure()
    fig1, axs1 = plt.subplots(1,1, figsize=(10,6), constrained_layout=True)
    axs1.plot(np.mean(prag_cont_array[1], axis = 0), linewidth=linewidth, color = 'tab:red')
    axs1.plot(np.mean(human_array[1], axis = 0), linewidth=linewidth, color = 'tab:orange')
    axs1.plot(np.mean(imt_array[1], axis = 0), linewidth=linewidth, color = 'tab:blue')
    axs1.plot(np.mean(imt_human_array[1], axis = 0), linewidth=linewidth, color = 'tab:green')
    axs1.plot(np.mean(rand_ital_array[1], axis = 0), linewidth=linewidth, color = 'tab:purple')

    axs1.set_title('L2 Distance', fontweight="bold", size=29)
    idx = 1
    x_range = 31

    m, l, h = mean_confidence_interval(prag_cont_array[idx], confidence=confidence)
    axs1.fill_between(np.arange(x_range), l, h,
                alpha=0.2, facecolor='tab:red', edgecolor=(0.8392156862745098, 0.15294117647058825, 0.1568627450980392,0.5)) 
    m, l, h = mean_confidence_interval(human_array[idx], confidence=confidence)
    axs1.fill_between(np.arange(x_range), l, h,
                 alpha=0.2, facecolor='tab:orange', edgecolor=(1.0, 0.4980392156862745, 0.054901960784313725,0.5)) 
    m, l, h = mean_confidence_interval(imt_array[idx], confidence=confidence)
    axs1.fill_between(np.arange(x_range), l, h,
                 alpha=0.2, facecolor='tab:blue', edgecolor=(0.12156862745098039, 0.4666666666666667, 0.7058823529411765,0.5))
    m, l, h = mean_confidence_interval(imt_human_array[idx], confidence=confidence)
    axs1.fill_between(np.arange(x_range), l, h,
                 alpha=0.2, facecolor='tab:green', edgecolor=(0.17254901960784313, 0.6274509803921569, 0.17254901960784313,0.5)) 
    m, l, h = mean_confidence_interval(rand_ital_array[idx], confidence=confidence)
    axs1.fill_between(np.arange(x_range), l, h,
                 alpha=0.2, facecolor='tab:brown', edgecolor=(0.5903921568627451, 0.403921568627451, 0.7411764705882353)) 
    axs1.set_xlabel('Training Iteration', fontweight="bold", size=29)
    plt.savefig('data/map%d-main.pdf' % (map_num), dpi=300) 

def histogram(seed_low, seed_high, map_num):
    prag_cont = np.load("data/map%d/machine_ital.npy"% map_num, allow_pickle = True)[()]
    imt = np.load("data/map%d/machine_imt.npy"% map_num, allow_pickle = True)[()]
    human = np.load("data/map%d/human_ital.npy"% map_num, allow_pickle = True)[()]
    imt_human = np.load("data/map%d/human_imt.npy"% map_num, allow_pickle = True)[()]

    human_acc = [[],[],[],[]]
    imt_human_acc = [[],[],[],[]]
    prag_cont_acc = [[],[],[],[]]
    imt_acc = [[],[],[],[]]
    for i in range(4):
        for j in range(prag_cont[i].shape[0]):
            human_acc[i].append(human[i][j][-1]-human[i][j][0]) 
            imt_human_acc[i].append(imt_human[i][j][-1]-imt_human[i][j][0]) 
            prag_cont_acc[i].append(prag_cont[i][j][-1]-prag_cont[i][j][0]) 
            imt_acc[i].append(imt[i][j][-1]-imt[i][j][0])

    colors = ['orange', 'blue', 'red', 'green']

    num_bins = 10
    fig, axs = plt.subplots(2,2, figsize=(10,10), constrained_layout=True)
    axs[0, 0].hist([human_acc[0],imt_human_acc[0],prag_cont_acc[0],imt_acc[0]], num_bins, histtype='bar', color=colors)
    axs[0, 0].set_title('action prob total variation distance')

    labels = ['ITAL, Human', 'IMT, Human', 'ITAL', 'IMT'] 

    axs[0, 1].hist([human_acc[1],imt_human_acc[1],prag_cont_acc[1],imt_acc[1]], num_bins, histtype='bar', label=labels, color=colors)
    axs[0, 1].legend()
    axs[0, 1].set_title('reward param l2 dist')

    axs[1, 0].set_title('q l2 dist')
    axs[1, 0].hist([human_acc[2],imt_human_acc[2],prag_cont_acc[2],imt_acc[2]], num_bins, histtype='bar', color=colors)

    axs[1, 1].set_title('actual rewards (every 2 iter)')
    axs[1, 1].hist([human_acc[3],imt_human_acc[3],prag_cont_acc[3],imt_acc[3]], num_bins, histtype='bar', color=colors)

    plt.suptitle("Map %d" % (map_num))
    plt.savefig('hist%d.png' % (map_num))    
    plt.show()


def mean_confidence_interval(data, confidence=0.7):
    data = np.array([data[i] for i in range(data.shape[0])])
    n = len(data)
    m, se = np.mean(data, axis=0), scipy.stats.sem(data)
    h = se * 1#scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def boxplot():
    maps = [0, 1, 3, 4, 7]
    dfs = []
    for i in range(4):
        map_col = []
        datatype = []  
        data = []        
        for map_num in maps:
            prag_cont = np.load("data/map%d/machine_ital.npy"% map_num, allow_pickle = True)[()]
            imt = np.load("data/map%d/machine_imt.npy"% map_num, allow_pickle = True)[()]
            human = np.load("data/map%d/human_ital.npy"% map_num, allow_pickle = True)[()]
            imt_human = np.load("data/map%d/human_imt.npy"% map_num, allow_pickle = True)[()]

            for j in range(prag_cont[i].shape[0]):
                data.append(human[i][j][-1]-human[i][j][0]) 
                map_col.append(map_num)
                datatype.append("human_ital")
                data.append(imt_human[i][j][-1]-imt_human[i][j][0])
                map_col.append(map_num)
                datatype.append("human_imt")
                data.append(prag_cont[i][j][-1]-prag_cont[i][j][0]) 
                map_col.append(map_num)
                datatype.append("machine_ital")                                                 
                data.append(imt[i][j][-1]-imt[i][j][0])
                map_col.append(map_num)
                datatype.append("machine_imt")     
        df = pd.DataFrame({'map_col': map_col,
                       'datatype': datatype,
                       'data': data})    
        dfs.append(df)  
    palette = {"human_ital":sns.xkcd_rgb["red"],"human_imt":sns.xkcd_rgb["burnt orange"], \
                "machine_ital":sns.xkcd_rgb["orange"], 'machine_imt': sns.xkcd_rgb['green']}   
    fig, axs = plt.subplots(2,2, figsize=(10,10), constrained_layout=True)
    plt1 = sns.boxplot(x="map_col", y="data",
                 hue="datatype", data=dfs[0], ax=axs[0, 0], palette=palette)
    sns.boxplot(x="map_col", y="data",
                 hue="datatype", data=dfs[1], ax=axs[0, 1], palette=palette)
    plt2 = sns.boxplot(x="map_col", y="data",
                 hue="datatype", data=dfs[2], ax=axs[1, 0], palette=palette)  
    plt3 = sns.boxplot(x="map_col", y="data",
                 hue="datatype", data=dfs[3], ax=axs[1, 1], palette=palette)
    axs[0, 0].set_title('action prob total variation distance')    
    axs[0, 1].set_title('reward param l2 dist')
    axs[1, 0].set_title('q l2 dist')
    axs[1, 1].set_title('actual rewards (every 2 iter)')
    plt1.legend_.remove()
    plt2.legend_.remove()
    plt3.legend_.remove()
    plt.savefig('boxplot.png')    
    plt.show()

def barplot():
    axis_title = {0: 'Total Variation', 1: 'L2 Distance', 2: 'Actual Reward'}
    datatype = OrderedDict({0: 'human_ital', 1: 'human_imt', 2: 'machine_ital', 3: 'machine_imt', 4: 'rand_ital'})
    maps = OrderedDict({0: 0, 1: 1, 2: 3, 3: 4, 4: 7})
    colors = {0: 'tab:orange', 1: 'tab:green', 2: 'tab:red', 3: 'tab:blue', 4: 'tab:purple'}
    yrange = {0: [-0.009, 0.04], 1: [-0.22, 1.0], 2: [-100, 460]}
    plt.rc('xtick', labelsize=21) 
    plt.rc('ytick', labelsize=21) 
    fig, axs = plt.subplots(1,3, figsize=(37,12), constrained_layout=True)
    ind = np.arange(5)*2.5
    width = 0.47
    error_kwdict = {'marker': 'x', 'capsize': 2.5}
    for i in [0, 1, 3]:
        idx = i if i != 3 else 2
        t_data = ttest(i)
        data = []        
        for map_num in maps:
            data.append([])
            for t in datatype:
                category = np.load("data/map%d/%s.npy"% (maps[map_num], datatype[t]), allow_pickle = True)[()]
                temp = []
                for j in range(category[i].shape[0]):
                    if i == 3:
                        temp.append(category[i][j][-1]-category[i][j][0])
                    else:
                        temp.append(category[i][j][0]-category[i][j][1])
                data[map_num].append([np.mean(temp), scipy.stats.sem(temp)])            
        
        for t in datatype:
            rects = axs[idx].bar(ind+width*t, np.array(data)[:,t,0], width, yerr=np.array(data)[:,t,1], color=colors[t], label=datatype[t],error_kw=error_kwdict)     
            if datatype[t] == 'machine_ital':
                pvalues = [t['machine'] for t in t_data]
                autolabel(rects, axs[idx], np.array(data)[:,t,1], pvalues)
            if datatype[t] == 'human_ital':
                pvalues = [t['human'] for t in t_data]
                autolabel(rects, axs[idx], np.array(data)[:,t,1], pvalues)
            #pdb.set_trace()
            axs[idx].set_xticks(ind + width*1.9)
            axs[idx].set_xticklabels(('Map A', 'Map B', 'Map C', 'Map D', 'Map E'))
            axs[idx].set_ylim([yrange[idx][0],yrange[idx][1]])
            axs[idx].set_title(axis_title[idx], fontweight="bold", size=29)
        
    plt.savefig('data/barplot.pdf' , dpi=300) 


def autolabel(rects, ax, yerr_heights, pvalues, previousH=None):
    """Attach a text label above each bar in *rects*, displaying its height."""
    arrowprops = {'arrowstyle': '-['}
    assert(len(rects) == len(pvalues))
    heights = []
    for i in range(len(rects)):
        if not previousH:
            height = rects[i].get_height() + yerr_heights[i] + 0.001
        else:
            height = previousH[i]
        heights.append(height)
        print(height)
        ax.annotate('{:.0e}'.format(pvalues[i]),
                    xy=(rects[i].get_x() + rects[i].get_width(), height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points", arrowprops=arrowprops,
                    ha='center', va='bottom', fontweight="bold", size=27)
    return heights

def ttest(category_idx):
    maps = OrderedDict({0: 0, 1: 1, 2: 3, 3: 4, 4: 7})
    datatype = OrderedDict({0: 'human_ital', 1: 'human_imt', 2: 'machine_ital', 3: 'machine_imt'})
    t_data = []
    for map_num in maps:

        human_ital = np.load("data/map%d/%s.npy"% (maps[map_num], datatype[0]), allow_pickle = True)[()]
        human_imt = np.load("data/map%d/%s.npy"% (maps[map_num], datatype[1]), allow_pickle = True)[()]
        machine_ital = np.load("data/map%d/%s.npy"% (maps[map_num], datatype[2]), allow_pickle = True)[()]
        machine_imt = np.load("data/map%d/%s.npy"% (maps[map_num], datatype[3]), allow_pickle = True)[()]
        assert(human_ital[category_idx].shape[0]==human_imt[category_idx].shape[0])

        temp_ital, temp_imt, temp_ital_machine, temp_imt_machine = [], [], [], []
        for player_idx in range(human_ital[category_idx].shape[0]):
            temp_ital.append(human_ital[category_idx][player_idx][-1] - human_ital[category_idx][player_idx][0])
            temp_imt.append(human_imt[category_idx][player_idx][-1] - human_imt[category_idx][player_idx][0])
            temp_ital_machine.append(machine_ital[category_idx][player_idx][-1] - machine_ital[category_idx][player_idx][0])
            temp_imt_machine.append(machine_imt[category_idx][player_idx][-1] - machine_imt[category_idx][player_idx][0])
        #pdb.set_trace()
        map_data = {'human': scipy.stats.ttest_rel(temp_ital, temp_imt)[1]/2,
                    'machine': scipy.stats.ttest_rel(temp_ital_machine, temp_imt_machine)[1]/2}
        t_data.append(map_data)
    return t_data

def ttest_plot(category_idx=0):
    axis_title = {0: 'Total Variation', 1: 'L2 Distance', 3: 'Actual Reward'}

    datatype = OrderedDict({0: 'human_ital', 1: 'human_imt', 2: 'machine_ital', 3: 'machine_imt'})
    maps = OrderedDict({0: 0, 1: 1, 2: 3, 3: 4, 4: 7})
    colors = {0: 'tab:orange', 1: 'tab:green', 2: 'tab:red', 3: 'tab:blue'}
    yrange = {0: [-0.0225, 0.05], 1: [-0.47, 1], 3: [-30, 450]}
    plt.rc('xtick', labelsize=20) 
    plt.rc('ytick', labelsize=20) 
    fig, axs = plt.subplots(2,1, figsize=(10,20), constrained_layout=True)
    ind = np.arange(5)*2
    width = 0.47
    error_kwdict = {'marker': 'x', 'capsize': 2.5}

    data = []      
    for map_num in maps:
        data.append([])
        for t in datatype:
            category = np.load("data/map%d/%s.npy"% (maps[map_num], datatype[t]), allow_pickle = True)[()]
            temp = []
            for j in range(category[category_idx].shape[0]):
                temp.append(np.abs(category[category_idx][j][0]-category[category_idx][j][-1]))
            
            data[map_num].append([np.mean(temp), scipy.stats.sem(temp)])            
    for t in datatype:
        axs[0].bar(ind+width*t, np.array(data)[:,t,0], width, yerr=np.array(data)[:,t,1], color=colors[t], label=datatype[t],error_kw=error_kwdict)     
    axs[0].set_xticks(ind + width)
    axs[0].set_xticklabels(('Map A', 'Map B', 'Map C', 'Map D', 'Map E'))
    axs[0].set_ylim([yrange[category_idx][0],yrange[category_idx][1]])
    axs[0].set_title(axis_title[category_idx], fontweight="bold", size=29)
    

    t_data = []
    for map_num in maps:
        map_data = []
        human_ital = np.load("data/map%d/%s.npy"% (maps[map_num], datatype[0]), allow_pickle = True)[()]
        human_imt = np.load("data/map%d/%s.npy"% (maps[map_num], datatype[1]), allow_pickle = True)[()]
        assert(human_ital[category_idx].shape[0]==human_imt[category_idx].shape[0])
        for iteration_idx in range(2, human_ital[category_idx][0].shape[0]):
            temp_ital, temp_imt = [], []
            for player_idx in range(human_ital[category_idx].shape[0]):
                temp_ital.append(human_ital[category_idx][player_idx][iteration_idx] - human_ital[category_idx][player_idx][0])
                temp_imt.append(human_imt[category_idx][player_idx][iteration_idx] - human_imt[category_idx][player_idx][0])
            #pdb.set_trace()
            map_data.append(scipy.stats.ttest_rel(temp_ital, temp_imt)[1]/2)
        axs[1].plot(map_data, label="Map %d" % maps[map_num])
        t_data.append(map_data)
    axs[1].plot(np.mean(t_data, axis=0), label="Average")
    axs[1].legend(fontsize=20)
    axs[1].set_xlabel('Iteration', size=20)
    plt.savefig('data/ttest%s.pdf' % (axis_title[category_idx]), dpi=300) 

def barplot_SC(category_idx=0):
    axis_title = {0: 'Total Policy Variance', 1: 'L2 Distance', 2: 'Actual Reward'}
    file_title = {0: 'totalVariance', 1: 'l2', 2: 'rewards'}
    datatype = OrderedDict({0: 'human_ital', 1: 'human_imt', 2: 'machine_ital', 3: 'machine_imt', 4: 'rand_ital'})
    datatype_l = OrderedDict({0: 'Human ITAL', 1: 'Human IMT', 2: 'Machine ITAL', 3: 'Machine IMT', 4: 'Random ITAL'})
    maps = OrderedDict({0: 0, 1: 1, 2: 3, 3: 4, 4: 7})
    colors = {0: 'tab:orange', 1: 'tab:green', 2: 'tab:red', 3: 'tab:blue', 4: 'tab:purple'}
    yrange = {0: [-0.009, 0.04], 1: [-0.20, 1.2], 2: [-100, 460]}
    plt.rc('xtick', labelsize=35) 
    plt.rc('ytick', labelsize=25) 
    fig, axs = plt.subplots(1,1, figsize=(20,12.5), constrained_layout=True)
    ind = np.arange(5)
    width = 0.15
    error_kwdict = {'marker': 'x', 'capsize': 2.5}

    idx = category_idx if category_idx != 3 else 2
    t_data = ttest(category_idx)
    data = []        
    for map_num in maps:
        data.append([])
        for t in datatype:
            category = np.load("data/map%d/%s.npy"% (maps[map_num], datatype[t]), allow_pickle = True)[()]
            temp = []
            for j in range(category[category_idx].shape[0]):
                if category_idx == 3:
                    temp.append(category[category_idx][j][-1]-category[category_idx][j][0])
                else:
                    temp.append(category[category_idx][j][0]-category[category_idx][j][1])
            data[map_num].append([np.mean(temp), scipy.stats.sem(temp)])            
    
    for t in [2,0,1,3,4]:
        rects = axs.bar(ind+width*t, np.array(data)[:,t,0], width, yerr=np.array(data)[:,t,1], color=colors[t], label=datatype_l[t],error_kw=error_kwdict)     
        if datatype[t] == 'machine_ital':
            pvalues = [t['machine'] for t in t_data]
            heights = autolabel(rects, axs, np.array(data)[:,t,1], pvalues)
        if datatype[t] == 'human_ital':
            pvalues = [t['human'] for t in t_data]
            _ = autolabel(rects, axs, np.array(data)[:,t,1], pvalues, heights)
        #pdb.set_trace()
    axs.set_xticks(ind + width*1.9)
    axs.set_xticklabels(('Map A', 'Map B', 'Map C', 'Map D', 'Map E'), fontweight="bold")
    axs.set_ylim([yrange[idx][0],yrange[idx][1]])
    axs.set_title(axis_title[idx], fontweight="bold", size=39)
    col = 2 if (category_idx == 0 or category_idx == 1) else 1
    l = axs.legend(fontsize=29, ncol=col)
    for i, text in enumerate(l.get_texts()):
        text.set_weight("bold")  
    plt.savefig('data/human_%s.pdf' % file_title[idx], dpi=300) 

if __name__ == '__main__':
    print('Collecting all data...')
    for map_num in [0,1,3,4,7]:
        CollectData(41, 100, map_num)
    
    print('Collecting all plots...')
    
    with plt.style.context('seaborn-darkgrid'):
        for map_id_ in [0,1,3,4,7]:
            for c in [0,1,3]:
                plot_average_SC(map_id_, c)
    
    with plt.style.context('seaborn-darkgrid'):
        barplot_SC(map_id)
    