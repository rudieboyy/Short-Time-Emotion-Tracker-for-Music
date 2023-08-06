from __future__ import print_function, division, absolute_import, unicode_literals
import six
import os

import numpy as np
import torch
import torch.nn as nn
import merAE_v
import audio_processor as ap
import sys
import pickle
from collections import defaultdict
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

class DataforTorch(torch.utils.data.Dataset):
	def __init__(self,path):
		super(DataforTorch,self).__init__()
		self.path = path
		with open(self.path, "rb") as fp:
			self.temp = pickle.load(fp)

	def __getitem__(self,index):
		resultDict = self.temp[index]
		
		if torch.max(torch.abs(resultDict['arousal'])) > 1 or torch.max(torch.abs(resultDict['valence'])) > 1 :
			print('problem')
			a=bbb
		resultDict['audio'] = resultDict['audio'].squeeze(0)
		resultDict['arousal']= resultDict['arousal'].squeeze(0)
		resultDict['valence']= resultDict['valence'].squeeze(0)
		return resultDict

	def __len__(self):
		return len(self.temp)

if __name__ =="__main__":
	batch_size = 1
	demo_v = np.zeros(60)
	dataset = DataforTorch("Testset")
	dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size,shuffle=False, num_workers = 2)

	model=merAE_v.MerAE_V(batch_size)
	model.load_state_dict(torch.load("AEModel_v.pt",map_location=lambda storage,loc:storage))

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	model.eval()

	criterion = nn.MSELoss()

	metricsDict = defaultdict(list)
	diterator = iter(dataloader)
	with trange(len(dataloader)) as t:
		for idx in t:
			sample = next(diterator)
			audio = sample['audio'].to(device).type(torch.FloatTensor)
			valence = sample['valence'].to(device).type(torch.FloatTensor)
			with torch.no_grad():
				v_hat = model(audio)
				loss = criterion(v_hat,valence)
				if idx == 31:
					demo_v = v_hat
			loss = loss.detach().cpu().numpy()
			metricsDict['loss'].append(loss.item())
	

	ValAVG = np.mean(metricsDict['loss'])
	ValMIN = np.amin(metricsDict['loss'])
	ValMINX = np.argmin(metricsDict['loss'])
	# print("31song's loss is:%.5f\n"%(metricsDict['loss'][31]))
	print ("ValAVG:%.5f\n" %(ValAVG))
	print ("ValMIN:%.5f\n" %(ValMIN))
	print ("ValMINX: " ,ValMINX)
	demo_v =demo_v.detach().cpu().numpy()
	np.set_printoptions(precision=5,suppress=True)
	print("Best valence predications are ", demo_v)
	plt.figure()
	plt.plot(metricsDict['loss']) 
	plt.title("valence loss test set")
	plt.xlabel('#songs')
	plt.ylabel('MSE loss')

	plt.savefig("val_losstest.png")
	plt.gcf().clear()