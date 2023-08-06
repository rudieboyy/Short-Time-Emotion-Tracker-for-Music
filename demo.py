from __future__ import print_function, division, absolute_import, unicode_literals
import six
import os

import numpy as np
import torch
import torch.nn as nn
import merAE_a
import merAE_v
import sys
import matplotlib.pyplot as plt
import librosa
from scipy.misc import imread
from scipy.io import wavfile
import pyaudio
import wave 

if __name__ =="__main__":
	batch_size = 1
	demo_a = np.zeros(60)
	demo_v = np.zeros(60)
	# audio processing
	logam = librosa.power_to_db
	melgram = librosa.feature.melspectrogram
	audio_file = '216.wav'
	x, SR = librosa.load(audio_file)
	X = logam(melgram(y=x, sr=SR, hop_length=647,n_fft=1323, n_mels=96,power = 1.0)**2)
	X_Mel = np.zeros((96,60))
	print(X.shape)
	for i in range(60):
		X_Mel[:,i] = X[:,i*17:(i*17+16)].mean(1) 
	X_Mel[:,59] = X[:,1020:].mean(1)
	X_Mel = X_Mel[np.newaxis,np.newaxis,:,:]
	##arousal
	model_a=merAE_a.MerAE_A(batch_size)
	model_a.load_state_dict(torch.load("AEModel_a.pt",map_location=lambda storage,loc:storage))

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model_a.to(device)
	model_a.eval()
	audio = torch.from_numpy(X_Mel).float().to(device)
	a_hat = model_a(audio)
	demo_a = a_hat.detach().cpu().numpy()
	##valence
	model_v =merAE_v.MerAE_V(batch_size)
	model_v.load_state_dict(torch.load("AEModel_v.pt",map_location=lambda storage,loc:storage))

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model_v.to(device)
	model_v.eval()
	v_hat = model_v(audio)
	demo_v = v_hat.detach().cpu().numpy()

	# np.set_printoptions(precision=5,suppress=True)
	# print("arousals are ", demo_a)

	# np.set_printoptions(precision=5,suppress=True)
	# print("valences are ", demo_v)

	im = imread("resized_VA.gif")
	plt.figure()
	xmin, xmax, ymin, ymax = (-1,1,-1,1)
	aspect = im.shape[0] / im.shape[1] * (xmax - xmin)/(ymax - ymin)
	plt.imshow(im,extent=[xmin, xmax, ymin, ymax], aspect=aspect)
	chunk = 11025
	f = wave.open(audio_file,'rb')
	p = pyaudio.PyAudio()

	stream = p.open(format = p.get_format_from_width(f.getsampwidth()),
					channels = f.getnchannels(),
					rate = f.getframerate(),
					output = True)
	data = f.readframes(chunk)
	i = 0
	plt.ion()
	plt.show()
	while data:
		stream.write(data)
		data = f.readframes(chunk)
		plt.plot(demo_a[0][i],demo_v[0][i],"go",markersize = 3.5 )
		plt.draw()
		plt.pause(0.00001)
		i +=1

	stream.stop_stream()
	stream.close()
	p.terminate()

	# plt.figure()
	# xmin, xmax, ymin, ymax = (-1,1,-1,1)
	# aspect = im.shape[0] / im.shape[1] * (xmax - xmin)/(ymax - ymin)
	# plt.imshow(im,extent=[xmin, xmax, ymin, ymax], aspect=aspect)
	# plt.plot(demo_a,demo_v,"go",markersize = 3.5 )
	# plt.show()
	# plt.axis([xmin, xmax, ymin, ymax]);
