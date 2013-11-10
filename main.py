from ML_LAB_1 import *

# Load audio sources
source_files = ['beet.wav', 'beet9.wav', 'beet92.wav', 'mike.wav', 'street.wav']
wav_data = []
sample_rate = None
for f in source_files:
    sr, data = scipy.io.wavfile.read(f)
    if sample_rate is None:
        sample_rate = sr
    else:
        assert(sample_rate == sr)
    wav_data.append(data[:190000])  # cut off the last part so that all signals have same length

# Create source and measurement data
S = np.c_[wav_data]
plot_signals(S)
X = make_mixtures(S)
plot_signals(X)
# Save mixtures to disk, so you can listen to them in your audio player
for i in range(X.shape[0]):
   save_wav(X[i, :], 'X' + str(i) + '.wav', sample_rate)

# Run ICA
wX = whiten(X)
W = plotICA(wX,a0,0.01,0.1,20)
a = dot(W,wX)
rcParams['figure.figsize'] = 10, 5
plot_signals(a)
for i in range(a.shape[0]):
   save_wav(a[i, :], 'A' + str(i) + '.wav', sample_rate)

