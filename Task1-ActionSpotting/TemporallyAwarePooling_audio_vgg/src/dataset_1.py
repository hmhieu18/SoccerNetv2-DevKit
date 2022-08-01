from torch.utils.data import Dataset

import numpy as np
import random
import os
import time


from tqdm import tqdm

import torch

import logging
import json

from SoccerNet.Downloader import getListGames
from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.Evaluation.utils import AverageMeter, EVENT_DICTIONARY_V2, INVERSE_EVENT_DICTIONARY_V2
from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V1, INVERSE_EVENT_DICTIONARY_V1


def getShapeWithoutLoading(numpyFile):
    with open(numpyFile, 'rb') as f:
        major, minor = np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
    return shape


def feats2clip(feats, stride, clip_length, padding="replicate_last", off=0):
    if padding == "zeropad":
        print("beforepadding", feats.shape)
        pad = feats.shape[0] - int(feats.shape[0]/stride)*stride
        print("pad need to be", clip_length-pad)
        m = torch.nn.ZeroPad2d((0, 0, clip_length-pad, 0))
        feats = m(feats)
        print("afterpadding", feats.shape)
        # nn.ZeroPad2d(2)

    idx = torch.arange(start=0, end=feats.shape[0]-1, step=stride)
    idxs = []
    for i in torch.arange(-off, clip_length-off):
        idxs.append(idx+i)
    idx = torch.stack(idxs, dim=1)

    if padding == "replicate_last":
        idx = idx.clamp(0, feats.shape[0]-1)
    # print(idx)
    return feats[idx, ...]
# zero pad to the end of the clip


def padding(feats, shape):
    if(feats.shape < shape):
        result = np.zeros(shape)
        result[:feats.shape[0],
               :feats.shape[1]] = feats
    else:
        result = feats[0:shape[0], :]
    return result


class SoccerNetClips(Dataset):
    def __init__(self, visual_path, audio_path, visual_features="ResNET_PCA512.npy", audio_features="224ppyAudioAnalysis.npy", split=["train"], version=1,
                 framerate=2, window_size=15, listGames=None, ):
        self.visual_path = visual_path
        self.audio_path = audio_path

        self.listGames = listGames

        self.visual_features = visual_features
        self.audio_feaures = audio_features

        self.window_size_frame = window_size*framerate
        self.version = version
        if version == 1:
            self.num_classes = 3
            self.labels = "Labels.json"
        elif version == 2:
            self.dict_event = EVENT_DICTIONARY_V2
            self.num_classes = 17
            self.labels = "Labels-v2.json"

        # logging.info("Checking/Download features and labels locally")
        # downloader = SoccerNetDownloader(visual_path)
        # downloader.downloadGames(files=[
        #                          self.labels, f"1_{self.visual_features}", f"2_{self.visual_features}"], split=split, verbose=False, randomized=True)

        # logging.info("Pre-compute clips")

        self.game_feats = list()
        self.game_audio_feats = list()
        self.game_labels = list()

        # game_counter = 0
        for game in tqdm(self.listGames):
            # get filename
            feats1_filename = os.path.join(
                self.visual_path, game, f"1_{self.visual_features}")
            feats2_filename = os.path.join(
                self.visual_path, game, f"2_{self.visual_features}")
            audio_feats1_filename = os.path.join(
                self.audio_path, game, f"1_{self.audio_feaures}")
            audio_feats2_filename = os.path.join(
                self.audio_path, game, f"2_{self.audio_feaures}")

            feat_half1 = np.load(feats1_filename)
            feat_half2 = np.load(feats2_filename)

            audio_feat_half1 = np.load(audio_feats1_filename)
            audio_feat_half2 = np.load(audio_feats2_filename)



            feats1_shape = feat_half1.shape
            feats2_shape = feat_half2.shape

            labels = json.load(
                open(os.path.join(self.visual_path, game, self.labels)))

            label_half1 = np.zeros((feats1_shape[0], self.num_classes+1))
            label_half1[:, 0] = 1  # those are BG classes
            label_half2 = np.zeros((feats2_shape[0], self.num_classes+1))
            label_half2[:, 0] = 1  # those are BG classes
            for annotation in labels["annotations"]:

                time = annotation["gameTime"]
                event = annotation["label"]

                half = int(time[0])

                minutes = int(time[-5:-3])
                seconds = int(time[-2::])
                frame = framerate * (seconds + 60 * minutes)

                if version == 1:
                    if "card" in event:
                        label = 0
                    elif "subs" in event:
                        label = 1
                    elif "soccer" in event:
                        label = 2
                    else:
                        continue
                elif version >= 2:
                    if event not in self.dict_event:
                        continue
                    label = self.dict_event[event]

                # if label outside temporal of view
                if half == 1 and frame//self.window_size_frame >= label_half1.shape[0]:
                    continue
                if half == 2 and frame//self.window_size_frame >= label_half2.shape[0]:
                    continue

                if half == 1:
                    # not BG anymore
                    label_half1[frame//self.window_size_frame][0] = 0
                    # that's my class
                    label_half1[frame//self.window_size_frame][label+1] = 1

                if half == 2:
                    # not BG anymore
                    label_half2[frame//self.window_size_frame][0] = 0
                    # that's my class
                    label_half2[frame//self.window_size_frame][label+1] = 1
            self.game_feats.append(feat_half1)
            self.game_feats.append(feat_half2)
            
            self.game_audio_feats.append(audio_feat_half1)
            self.game_audio_feats.append(audio_feat_half2)

            self.game_labels.append(label_half1)
            self.game_labels.append(label_half2)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            clip_feat (np.array): clip of features.
            clip_labels (np.array): clip of labels for the segmentation.
            clip_targets (np.array): clip of targets for the spotting.
        """
        # print("index", index)
        # print("game_feats", self.game_feats[index].shape)
        # print(self.game_audio_feats_files[index], np.load(self.game_audio_feats_files[index]).shape)
        # print(self.game_feats_files[index], np.load(self.game_feats_files[index]).shape)

        return self.game_feats[index], self.game_audio_feats[index], self.game_labels[index]
        # return self.game_feats[index, :, :], self.game_labels[index, :]

    def __len__(self):
        return len(self.game_feats)


class SoccerNetClipsTesting(Dataset):
    def __init__(self, visual_path, audio_path, features="ResNET_PCA512.npy", audio_features="224p_VGGish_Test", split=["test"], version=1,
                 framerate=2, window_size=15, listGames=None,):
        # self.path = path
        self.visual_path = visual_path
        self.audio_path = audio_path

        self.listGames = listGames
        self.features = features
        self.audio_features = audio_features
        self.window_size_frame = window_size*framerate
        self.framerate = framerate
        self.version = version
        self.split = split
        if version == 1:
            self.dict_event = EVENT_DICTIONARY_V1
            self.num_classes = 3
            self.labels = "Labels.json"
        elif version == 2:
            self.dict_event = EVENT_DICTIONARY_V2
            self.num_classes = 17
            self.labels = "Labels-v2.json"

        # logging.info("Checking/Download features and labels locally")
        # downloader = SoccerNetDownloader(path)
        # for s in split:
        #     if s == "challenge":
        #         downloader.downloadGames(files=[f"1_{self.features}", f"2_{self.features}"], split=[
        #                                  s], verbose=False, randomized=True)
        #     else:
        #         downloader.downloadGames(files=[self.labels, f"1_{self.features}", f"2_{self.features}"], split=[
        #                                  s], verbose=False, randomized=True)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            feat_half1 (np.array): features for the 1st half.
            feat_half2 (np.array): features for the 2nd half.
            label_half1 (np.array): labels (one-hot) for the 1st half.
            label_half2 (np.array): labels (one-hot) for the 2nd half.
        """
        game = self.listGames[index]
        feats1_filename = os.path.join(
            self.visual_path, game, f"1_{self.visual_features}")
        feats2_filename = os.path.join(
            self.visual_path, game, f"2_{self.visual_features}")
        audio_feats1_filename = os.path.join(
            self.audio_path, game, f"1_{self.audio_feaures}")
        audio_feats2_filename = os.path.join(
            self.audio_path, game, f"2_{self.audio_feaures}")

        feat_half1 = np.load(feats1_filename)
        feat_half2 = np.load(feats2_filename)
        
        audio_feat_half1 = np.load(audio_feats1_filename)
        audio_feat_half2 = np.load(audio_feats2_filename)


        return game, feat_half1, feat_half2, audio_feat_half1, audio_feat_half2

    def __len__(self):
        return len(self.listGames)
