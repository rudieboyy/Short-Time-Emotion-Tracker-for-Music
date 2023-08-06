from __future__ import print_function, division, absolute_import, unicode_literals
import six
import os

import numpy as np
import torch
import torch.nn as nn
import merAE_a
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
	demo_a = np.zeros(60)
	dataset = DataforTorch("Testset")
	dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size,shuffle=False, num_workers = 2)

	model=merAE_a.MerAE_A(batch_size)
	model.load_state_dict(torch.load("AEModel_a.pt",map_location=lambda storage,loc:storage))

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
			arousal = sample['arousal'].to(device).type(torch.FloatTensor)
			with torch.no_grad():
				a_hat = model(audio)
				loss = criterion(a_hat,arousal)
				if idx == 31:
					demo_a = a_hat
			loss = loss.detach().cpu().numpy()
			metricsDict['loss'].append(loss.item())
	
	ArsAVG = np.mean(metricsDict['loss'])
	ArsMIN = np.amin(metricsDict['loss'])
	ArsMINX = np.argmin(metricsDict['loss'])
	# # print("86ong's loss is:%.5f\n"%(metricsDict['loss'][86]))
	print ("ArsVG:%.5f\n" %(ArsAVG))
	print ("ArsMIN:%.5f\n" %(ArsMIN))
	print ("ArsMINX: " ,ArsMINX)
	demo_a =demo_a.detach().cpu().numpy()
	np.set_printoptions(precision=5,suppress=True)
	print("Best arousal predications are ", demo_a)
	plt.figure()
	plt.plot(metricsDict['loss']) 
	plt.title("arousal loss test set")
	plt.xlabel('#songs')
	plt.ylabel('MSE loss')

	plt.savefig("ars_losstest.png")
	plt.gcf().clear()