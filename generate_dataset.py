#
# @author: Yoon mo Yang 
#
from __future__ import print_function, division, absolute_import, unicode_literals
import six

import os
import librosa
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import audio_processor as ap

import torch
import torch.utils.data
import torchvision.transforms as transforms 
import pickle
'''
	blockSize = 1323 # 0.06sec = 60ms: 1323 samples with sr = 22.05 kHz
	hopSize = 647 # 0.029342 = 29.342ms: 647 samples => 51.1% overlap
	nMels = 96 # number of mel coeff (96)
'''

class generate_dataset(torch.utils.data.Dataset):
	def __init__(self, path, testSet = False, trainSet = False, valSet = False, mono = True, blockSize = 1323, hopSize = 647, nMels = 96):
		audioPaths = []
		if testSet: # for evaluation
			audioPath = os.path.join(path,'Test')
			arsPath = os.path.join(audioPath,'arousal_cont_average.csv')
			valPath = os.path.join(audioPath,'valence_cont_average.csv')
		if trainSet: # for training
			audioPath = os.path.join(path,'Train')
			arsPath = os.path.join(audioPath,'arousal_cont_average.csv')
			valPath = os.path.join(audioPath,'valence_cont_average.csv')
		if valSet:
			audioPath = os.path.join(path,'Val')
			arsPath = os.path.join(audioPath,'arousal_cont_average.csv')
			valPath = os.path.join(audioPath,'valence_cont_average.csv')

		for root, dirs, files in os.walk(audioPath):
			for f in files:
				mp3FilePath = os.path.join(root, f)
				if mp3FilePath.endswith('.mp3'):
					audioPaths.append(mp3FilePath)

		self.audioPaths = audioPaths
		self.ars_anno = pd.read_csv(arsPath)
		self.val_anno = pd.read_csv(valPath)
		self.testSet = testSet
		self.mono = mono

		self.blockSize = blockSize
		self.hopSize = hopSize
		self.nMels = nMels
	def __len__(self):
		return len(self.audioPaths)

	def __getitem__(self, idx):
		
		print(idx)
		if idx >= len(self.audioPaths):
			raise IndexError

		audio, fs = ap.mp3read(self.audioPaths[idx])
		ars_anno = np.array(self.ars_anno.iloc[idx,2:])
		val_anno = np.array(self.val_anno.iloc[idx,2:])

		if self.mono:
			#downmix here
			audio = np.mean(audio, axis=  -1)
		sample = {'audio':ap.compute_melgram(audio,blockSize = self.blockSize,hopSize = self.hopSize,nMels = self.nMels),'arousal':ars_anno,'valence':val_anno}

		return sample


if __name__ == "__main__":

	path_to_data = 'Emotion_in_Music_Database/clips_45seconds'
	
	dataset = generate_dataset(path_to_data, trainSet = True, mono = False)
	print(len(dataset))

	dataloader = torch.utils.data.DataLoader(dataset, batch_size = 1,shuffle=False, num_workers = 0)
	listOfDicts = []
	for idx, sample in enumerate(dataloader):
		# newidx = idx*32
		# if idx == 19:
		# 	f["audio"][newidx:,:,:,:] = sample['audio']
		# 	f["arousal"][newidx:,:] = sample['arousal']
		# 	f["valence"][newidx:,:] = sample['valence']
		# else:
		# 	f["audio"][newidx:newidx+32,:,:,:] = sample['audio']
		# 	f["arousal"][newidx:newidx+32,:] = sample['arousal']
		# 	f["valence"][newidx:newidx+32,:] = sample['valence']
		listOfDicts.append(sample)

	with open("TrainsetRNN", "wb") as fp:
			pickle.dump(listOfDicts, fp)



