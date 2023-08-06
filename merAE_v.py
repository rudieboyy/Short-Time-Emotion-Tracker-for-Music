#
# @author: Yoon mo Yang 
#
from __future__ import print_function, division, absolute_import, unicode_literals
import six
import os

import numpy as np
import pandas as pd
import h5py
import torch
import torchvision.transforms as transforms 
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from collections import defaultdict
import matplotlib.pyplot as plt
import audio_processor as ap
import pickle

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

class MerAE_V(nn.Module):
	def __init__(self, batch_size = 16):
		super(MerAE_V,self).__init__()
		self.batch_size = batch_size
		#########################################
        #        layers are defined below       #
        #########################################
        # convolutional layer 1 + maxpooling layer 1
		self.clayer1 = nn.Sequential(
			nn.Conv2d(1,16, kernel_size=(7,7),stride = (2,2), padding = (3,3)),
			nn.LeakyReLU()
		 	# nn.BatchNorm2d(16)
			)
		# convolutional layer 2 + maxpooling layer 2
		self.clayer2 = nn.Sequential(
			nn.Conv2d(16,64, kernel_size=(5,5),stride = (2,2), padding = (2,2)),
			nn.LeakyReLU()
			# nn.BatchNorm2d(64)
			)
		# Decoder
		self.clayer3 = nn.Sequential(
			nn.ConvTranspose2d(64,128, kernel_size=5,stride = (1,2),padding=(5,1)),
			nn.LeakyReLU()
			)

		self.clayer4 = nn.Sequential(
			nn.ConvTranspose2d(128,64, kernel_size=7,stride = (1,2),padding=(6,1)),
			nn.LeakyReLU()
			)

		self.clayer5 = nn.Sequential(
			nn.ConvTranspose2d(64,1, kernel_size=7,stride = 1,padding=(1,6)),
			nn.LeakyReLU()
			)
		self.maxp =nn.Sequential(
			nn.MaxPool2d(kernel_size = 1,stride = (14,2)),
			nn.Tanh()
			)

		self.initParams()

	def initParams(self):
		for param in self.parameters():
			if len(param.shape)>1:
				torch.nn.init.xavier_normal_(param)

	def encode(self,x):
		# print(x.shape)
		a = F.dropout(self.clayer1(x),p=0.75)
		# print(a.shape)
		b = F.dropout(self.clayer2(a),p=0.75)
		# print(b.shape)
		return b

	def decode(self,x):
		b = F.dropout(self.clayer3(x),p=0.75)
		# print(b.shape)
		a = F.dropout(self.clayer4(b),p=0.75)
		# print(a.shape)
		o = self.maxp(self.clayer5(a))
		# print(o.shape)
		o = o.view(-1, self.num_flat_features(o))
		# print(o.shsape)
		return o

	def forward(self,x):
		b = self.encode(x)
		o = self.decode(b)
		a_hat = o
		return a_hat
	def num_flat_features(self,x):
		size = x.size()[2:]
		num_features = 1
		for s in size:
			num_features *= s
		return num_features

if __name__ == "__main__":
	# For loss function
	eta = 0.999

	# PATH_DATASET = #path to hdf5 file (pre-processed data as a dictionary)


	#how many audio files to process fetched at each time: 32 files
	batch_size= 16

	#path to save the model
	savedFilename = "AEModel_v.pt"
	# savedFilename_v = "savedModel_MerSCNN_v.pt"

	dataset = DataforTorch("Trainset")
	dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size,shuffle=True, num_workers = 2)
	valset = DataforTorch("Valset")
	valloader = torch.utils.data.DataLoader(valset, batch_size = 1,shuffle=True, num_workers = 2)
    #initialize the Model
	model = MerAE_V(batch_size).cuda()
	# model_v = MerSCNN(batch_size).cuda()
    # model_val = MerCRNN(batch_size,hidden_size)#.cuda()
    
    #if you want to restore your previous saved model
	if os.path.exists(savedFilename):
		model.load_state_dict(torch.load(savedFilename))
	# if os.path.exists(savedFilename_v):	
		# model_v.load_state_dict(torch.load(savedFilename_v))

    #determine if cuda is available
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
	model.to(device)
	# model_v.to(device)
    # model_val.to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
	model.train(mode=True)
    #################################### 
    #Implement your training loop here
    #################################### 
	nepoc = 100
	# a_loss_sm = 0
	criterion = nn.MSELoss()
	criterionv = nn.MSELoss()
	# a_loss_trd = []
	epocMetrics = defaultdict(list)
	epocMetricsv = defaultdict(list)
	for epoc in range(nepoc):
        #Each time we fetch a batch of samples from the dataloader
		metricsDict = defaultdict(list)
		diterator = iter(dataloader)
		model.train(mode=True)
		with trange(len(dataloader)) as t:
			for idx in t:
				sample = next(diterator)

				model.zero_grad()

				audio = sample['audio'].to(device).type(torch.cuda.FloatTensor)
				valence = sample['valence'].to(device).type(torch.cuda.FloatTensor)

				v_hat = model(audio)

	            # loss function computation
				loss = criterion(v_hat,valence) # MSE loss
				loss.backward() # backward : backprop
				metricsDict['loss'].append(loss.item())
				optimizer.step() # Does the update

				t.set_description("lossAv: %.5f, lossCur: %.5f" % (np.mean(metricsDict['loss']), loss.item()))
		
		metricsDictv = defaultdict(list)
		viterator = iter(valloader)
		with trange(len(valloader)) as tv:
			for idx2 in tv:
				samplev = next(viterator)
				audiov = samplev['audio'].to(device).type(torch.cuda.FloatTensor)
				valencev = samplev['valence'].to(device).type(torch.cuda.FloatTensor)
				model.eval()
				with torch.no_grad():
					v_hatv = model(audiov)
					lossv = criterionv(v_hatv,valencev)
				lossv = lossv.detach().cpu().numpy()
				metricsDictv['lossv'].append(lossv.item())

			#save the model to savedFilename
			# torch.save(model_v.state_dict(),savedFilename_v)
		epocMetrics['loss'].append(np.mean(metricsDict['loss']))
		epocMetricsv['lossv'].append(np.mean(metricsDictv['lossv']))
		torch.save(model.state_dict(),savedFilename)
	plt.subplot(1,2,1)
	plt.plot(epocMetrics['loss']) 
	plt.title("valence loss training set")
	plt.xlabel('epoch number')
	plt.ylabel('MSE loss')

	plt.subplot(1,2,2)
	plt.plot(epocMetricsv['lossv'])
	plt.title("valence loss validation set")
	plt.xlabel('epoch number')


	plt.savefig("AEbest_v.png")
	plt.gcf().clear()



