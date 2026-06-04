import os
import random

import cv2
import numpy as np
import pandas as pd
import torch

from scipy.ndimage import rotate
from sklearn.preprocessing import StandardScaler
from utils import load_sequences,load_sequences_XSA

class XSADataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, clip_len=3, sampling_rate=1, num_clips=1, augment=False, frac=1.0, kinetics=False, n_TTA=0, label='Recur',n=9):

        self.split = split
        self.clip_len = clip_len
        self.num_clips = num_clips
        self.sampling_rate = sampling_rate
        self.augment = augment
        self.kinetics = kinetics
        self.label = label

        self.video_dir = os.path.join(data_dir, 'data')
        self.label_df = pd.read_excel(os.path.join(data_dir, self.split + '.xlsx'))
        self.CLASSES = ['No Recur', 'Recur']

        if n is not None:
            # self.label_df = self.label_df.iloc[:, :n]
            self.label_df['ID'] = self.label_df.ID.astype('str')
            self.label_df.iloc[:,3] = self.label_df.iloc[:,3].apply(lambda x: 1 if x == 1 else 0)
            self.label_df.iloc[:,5] = self.label_df.iloc[:,5].apply(lambda x: 1 if x == 1 else 0)
            self.label_df.iloc[:,7] = self.label_df.iloc[:,7].apply(lambda x: 1 if x == 1 else 0)

        if frac != 1.0:
            study_ids = np.sort(self.label_df['ID'].unique())

            self.label_df = self.label_df[self.label_df['ID'].isin(np.random.choice(study_ids, size=int(frac*study_ids.size), replace=False))]

            print('Num studies:', int(frac*study_ids.size))
        # self.label_df['label'] = self.label_df['DIM'].apply(lambda x: 1 if x == 'Local Metastasis' else 0)
        # self.label_df['label'] = self.label_df.iloc[:,2].apply(lambda x: 1 if x == 1 else 0)

        # self.label_df.drop('姓名')
        print(self.label_df.iloc[:,7].value_counts()) # 3 for recur, 7 for LTP, 5 for metastasis

        # Kinetics-400 mean and std
        self.mean = np.array([0.43216, 0.394666, 0.37645])
        self.std = np.array([0.22803, 0.22145, 0.216989])

    def _sample_frames(self, x):
        if self.split == 'train':
            if x.shape[0] > self.clip_len*self.sampling_rate:
                start_idx = np.random.choice(x.shape[0]-self.clip_len*self.sampling_rate, size=1)[0]
                x = x[start_idx:(start_idx+self.clip_len*self.sampling_rate):self.sampling_rate]
                x = np.transpose(x, (3, 0, 1, 2))
            else:
                x = x[::self.sampling_rate]
                x = np.pad(x, ((0, self.clip_len-x.shape[0]), (0, 0), (0, 0), (0, 0)), mode='constant')
                x = np.transpose(x, (3, 0, 1, 2))
        else:
            if x.shape[0] >= self.clip_len*self.sampling_rate + self.num_clips:
                start_indices = np.arange(0, x.shape[0]-self.clip_len*self.sampling_rate, (x.shape[0]-self.clip_len*self.sampling_rate) // self.num_clips)[:self.num_clips]
                x = np.stack([x[start_idx:(start_idx+self.clip_len*self.sampling_rate):self.sampling_rate] for start_idx in start_indices], axis=0)
                x = np.transpose(x, (4, 0, 1, 2, 3))
            elif x.shape[0] > self.clip_len*self.sampling_rate:
                x = x[::self.sampling_rate]
                x = x[:self.clip_len]
                x = np.stack([x] * self.num_clips, axis=0)
                x = np.transpose(x, (4, 0, 1, 2, 3))
            else:
                x = x[::self.sampling_rate]
                x = np.pad(x, ((0, self.clip_len-x.shape[0]), (0, 0), (0, 0), (0, 0)), mode='constant')
                x = np.stack([x] * self.num_clips, axis=0)
                x = np.transpose(x, (4, 0, 1, 2, 3))

        return x

    def _augment(self, x):
        pad = 8

        l, h, w, c = x.shape
        temp = np.zeros((l, h + 2 * pad, w + 2 * pad, c), dtype=x.dtype)
        temp[:, pad:-pad, pad:-pad, :] = x
        i, j = np.random.randint(0, 2 * pad, 2)
        x = temp[:, i:(i + h), j:(j + w), :]

        if random.uniform(0, 1) > 0.5:
            # H flip
            x = np.stack([cv2.flip(frame, 1) for frame in x], axis=0)

        if random.uniform(0, 1) > 0.5:
            # Rotation
            angle = np.random.choice(np.arange(-10, 11), size=1)[0]

            x = np.stack([rotate(frame, angle, reshape=False) for frame in x], axis=0)

        return x

    def __len__(self):
        return self.label_df.shape[0]

    def __getitem__(self, idx):
        # plax_prob, fname, acc_num, _, video_num, label = self.label_df.iloc[idx, :]
        # acc_num,fname,Metastasis,MFS,LTP,LTPT,centre,event,TimeToEvent  = self.label_df.iloc[idx, :] ######
        acc_num,fname,centre,event,TimeToEvent,_,_,LTP,LTPT = self.label_df.iloc[idx, :9] ######
        # cli_vars = np.asarray(self.label_df.iloc[idx, 9:].values, dtype=np.float32).flatten()
        scaler = StandardScaler().fit(self.label_df.iloc[:, 9:])
        cli_vars = np.asarray(scaler.transform(self.label_df.iloc[idx, 9:].to_numpy().reshape(1, -1)).squeeze())

        ## Depend on how we want to load the data here
        # x = load_sequences(self.video_dir,str(acc_num),4)
        x = load_sequences_XSA(self.video_dir,str(acc_num),6)

        if self.augment:
            x = self._augment(x)

        ## Add those back in the future.
        # x = self._sample_frames(x)
        x = (x - x.min()) / (x.max() - x.min())

        if self.kinetics:
            if self.split == 'train':
                x -= self.mean.reshape(3, 1, 1, 1)
                x /= self.std.reshape(3, 1, 1, 1)
            else:
                x -= self.mean.reshape(3, 1, 1, 1, 1)
                x /= self.std.reshape(3, 1, 1, 1, 1)

        # y = np.array([event,TimeToEvent])
        y = np.array([LTP,LTPT])
        #
        # return {'x': torch.from_numpy(x).float(), 'y': torch.from_numpy(y).float(), 'acc_num': acc_num, 'video_num': video_num, 'plax_prob': plax_prob}
        return {'x': torch.from_numpy(x).float(), 'y': torch.from_numpy(y).float(),
                'ID': acc_num,'cli': torch.from_numpy(cli_vars).float()}
