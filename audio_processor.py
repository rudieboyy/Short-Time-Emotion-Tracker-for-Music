#
# @author: Yoon mo Yang 
#
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

SR = 22050

def mp3read(audioPaths):
    x, sr = librosa.load(audioPaths,sr=SR)

    if x.dtype != np.float32:
        x = x/np.iinfo(x.dtype).max
    return x,sr

def compute_melgram(x,blockSize, hopSize, nMels):
	''' Compute a mel-spectrogram and returns it in a shape of (96,60), where
    96 == #mel-bins and 60 == #time frame
    parameters
    ----------
    audio_path: path for the audio file.
                Any format supported by audioread will work.

    '''
	# mel-spec parameters
	# N_FFT = blockSize # 0.06sec = 60ms: 1323 samples with sr = 22.05 kHz
	# HOP_SIZE = hopSize # 0.029342 = 29.342ms: 647 samples => 51.1% overlap
	# N_MEL = nMels # number of mel coeff (96)

	# print type(x)
	logam = librosa.power_to_db
	melgram = librosa.feature.melspectrogram
	x = x[331104:] # disregard the first 15 seconds

	X = logam(melgram(y=x, sr=SR, hop_length=hopSize,n_fft=blockSize, n_mels=nMels,power = 1.0)**2) # (n_mels,t) = (96 mels,1021 frames)
    # take average over 17 frames so we get 60 frames in total.
	print(X.shape)
	X_Mel = np.zeros((96,60))
	if X.shape[1] == 1009:
		for i in range(60):
			X_Mel[:,i] = X[:,i*16:(i*16+15)].mean(1)
	elif X.shape[1] == 1020:
		for i in range(60):
			X_Mel[:,i] = X[:,i*17:(i*17+16)].mean(1) 
	else:
		for i in range(60):
			X_Mel[:,i] = X[:,i*17:(i*17+16)].mean(1) #from 0 to 1002 : 1003frames 

	if X.shape[1] == 1009:
		X_Mel[:,59] = X[:,960:].mean(1)
	if X.shape[1] > 1020:
		X_Mel[:,59] = X[:,1020:].mean(1) # from 1003 to last frame(1024):22frames -> X_Mel has 60 time frames now
	
	X_Mel = X_Mel[np.newaxis,:,:]
	
	return X_Mel

if __name__ == "__main__":

	x,sr = librosa.load('2.mp3',sr=SR)
	X = compute_melgram(x,1323,647,96)
	plt.figure(figsize=(10,4))
	librosa.display.specshow(X,y_axis='mel',fmax = SR/2, x_axis='time')
	plt.title("mel-spectrogram")
	plt.colorbar(format='%+2.0f dB')
	plt.savefig('Mel Spectrogram')
