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


def learn(teacher, learner, mode, init_ws, train_iter, test_set, teacher_rewards, data, random_prob = None, human = False):

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

def download_all(seed_low, seed_high, map_num):
    
    downloaded = []
    for s in range(seed_low, seed_high + 1):
        try:
            DataDownload(s, map_num)
            downloaded.append(s)
        except UnboundLocalError:
            continue
    return downloaded

def plot_average(seed_low, seed_high, map_num):
    downloaded = download_all(seed_low, seed_high, map_num)
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
    learner = LearnerIRL(map_l, config_L)

    data_ital = np.load("map_%d_data%d_ital.npy" % (map_num, downloaded[0]), allow_pickle = True)[()]
    data_imt = np.load("map_%d_data%d_imt.npy"  % (map_num, downloaded[0]), allow_pickle = True)[()] 
    init_ws = data_ital['ws'][0]

    human = learn(teacher, learner, '%s_cont' % config.mode, init_ws, config.train_iter, test_set, gt_r_param_tea, data_ital,  None, True)
    imt_human = learn(teacher, learner, '%s_cont' % config.mode, init_ws, config.train_iter, test_set, gt_r_param_tea, data_imt, 1, True)
    prag_cont = learn(teacher, learner, '%s_cont' % config.mode, init_ws, config.train_iter, test_set, gt_r_param_tea, data_ital, None)
    imt = learn(teacher, learner, '%s_cont' % config.mode, init_ws, config.train_iter, test_set, gt_r_param_tea, data_imt, 1)    
    
    human_acc = [human[i] for i in range(4)]
    imt_human_acc = [imt_human[i] for i in range(4)]
    prag_cont_acc = [prag_cont[i] for i in range(4)]
    imt_acc = [imt[i] for i in range(4)]

    for s in downloaded[1:]:
        data_ital = np.load("map_%d_data%d_ital.npy" % (map_num, s), allow_pickle = True)[()]
        data_imt = np.load("map_%d_data%d_imt.npy"  % (map_num, s), allow_pickle = True)[()] 
        init_ws = data_ital['ws'][0]

        human = learn(teacher, learner, '%s_cont' % config.mode, init_ws, config.train_iter, test_set, gt_r_param_tea, data_ital,  None, True)
        imt_human = learn(teacher, learner, '%s_cont' % config.mode, init_ws, config.train_iter, test_set, gt_r_param_tea, data_imt, 1, True)
        prag_cont = learn(teacher, learner, '%s_cont' % config.mode, init_ws, config.train_iter, test_set, gt_r_param_tea, data_ital, None)
        imt = learn(teacher, learner, '%s_cont' % config.mode, init_ws, config.train_iter, test_set, gt_r_param_tea, data_imt, 1)    
        for i in range(4):
            human_acc[i] += human[i]
            imt_human_acc[i] += imt_human[i]
            prag_cont_acc[i] += prag_cont[i]
            imt_acc[i] += imt[i]

    for i in range(4):        
        human_acc[i] = human_acc[i]/(seed_high + 1 - seed_low)
        imt_human_acc[i] = imt_human_acc[i]/(seed_high + 1 - seed_low)
        prag_cont_acc[i] = prag_cont_acc[i]/(seed_high + 1 - seed_low)
        imt_acc[i] = imt_acc[i]/(seed_high + 1 - seed_low)
    
    fig, axs = plt.subplots(2,2, figsize=(10,10), constrained_layout=True)
    axs[0, 0].plot(prag_cont_acc[0])
    axs[0, 0].plot(human_acc[0])
    axs[0, 0].plot(imt_acc[0])
    axs[0, 0].plot(imt_human_acc[0])

    axs[0, 0].set_title('action prob total variation distance')

    axs[0, 1].plot(prag_cont_acc[1], label='ITAL')
    axs[0, 1].plot(human_acc[1], label='ITAL, Human')
    axs[0, 1].plot(imt_acc[1], label='IMT')
    axs[0, 1].plot(imt_human_acc[1], label='IMT, Human')

    axs[0, 1].legend()

    axs[0, 1].set_title('reward param l2 dist')

    axs[1, 0].set_title('q l2 dist')
    axs[1, 0].plot(prag_cont_acc[2])
    axs[1, 0].plot(human_acc[2])
    axs[1, 0].plot(imt_acc[2])
    axs[1, 0].plot(imt_human_acc[2])

    axs[1, 1].plot(prag_cont_acc[3])
    axs[1, 1].plot(human_acc[3])
    axs[1, 1].plot(imt_acc[3])
    axs[1, 1].plot(imt_human_acc[3])

    axs[1, 1].set_title('actual rewards (every 2 iter)')
    axs[1, 1].plot([teacher_reward] * len(prag_cont_acc[3]))

    plt.suptitle("Map %d" % (map_num))
    plt.savefig('figure%d.png' % (map_num))    
    plt.show()

def histogram(seed_low, seed_high, map_num):
    downloaded = download_all(seed_low, seed_high, map_num)
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
    learner = LearnerIRL(map_l, config_L)
  
    human_acc = [[],[],[],[]]
    imt_human_acc = [[],[],[],[]]
    prag_cont_acc = [[],[],[],[]]
    imt_acc = [[],[],[],[]]

    for s in downloaded[0:]:
        data_ital = np.load("map_%d_data%d_ital.npy" % (map_num, s), allow_pickle = True)[()]
        data_imt = np.load("map_%d_data%d_imt.npy"  % (map_num, s), allow_pickle = True)[()] 
        init_ws = data_ital['ws'][0]

        human = learn(teacher, learner, '%s_cont' % config.mode, init_ws, config.train_iter, test_set, gt_r_param_tea, data_ital,  None, True)
        imt_human = learn(teacher, learner, '%s_cont' % config.mode, init_ws, config.train_iter, test_set, gt_r_param_tea, data_imt, 1, True)
        prag_cont = learn(teacher, learner, '%s_cont' % config.mode, init_ws, config.train_iter, test_set, gt_r_param_tea, data_ital, None)
        imt = learn(teacher, learner, '%s_cont' % config.mode, init_ws, config.train_iter, test_set, gt_r_param_tea, data_imt, 1)    
        for i in range(4):
            if i != 3:
                human_acc[i].append(human[i][-1]-human[i][0]) 
                imt_human_acc[i].append(imt_human[i][-1]-imt_human[i][0]) 
                prag_cont_acc[i].append(prag_cont[i][-1]-prag_cont[i][0]) 
                imt_acc[i].append(imt[i][-1]-imt[i][0])
            else:
                human_acc[i].append((human[i][-1]-human[i][0])[0]) 
                imt_human_acc[i].append((imt_human[i][-1]-imt_human[i][0])[0]) 
                prag_cont_acc[i].append((prag_cont[i][-1]-prag_cont[i][0])[0]) 
                imt_acc[i].append((imt[i][-1]-imt[i][0])[0])

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
    plt.savefig('figure%d.png' % (map_num))    
    plt.show()
if __name__ == '__main__':
    seed = int(sys.argv[1])
    map_id = int(sys.argv[2])
    d = DataDownload(seed, map_id)
    #d.graph_data()


