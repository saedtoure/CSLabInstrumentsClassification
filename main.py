import numpy as np
import pyaudio
import librosa.display, librosa
import matplotlib.pyplot as plt
from Predictor import Predictor
import peakutils.plot

# Parameters
# Signal Processing Parameters
fs = 44100         # Sampling Frequency
n_mels = 128       # Number of Mel bands
n_mfcc = 20       # Number of MFCCs

i=0
f,ax = plt.subplots(2)
f.canvas.set_window_title('Instruments Classification')
# Prepare the Plotting Environment with random starting values
x = np.arange(10000)
y = np.random.randn(10000)
#
# Plot 0 is for overtones series
li, = ax[0].plot(x,y)
ax[0].set_xlim(0,700)
ax[0].set_ylim(0,3)
ax[0].set_title("Overtones Series")
# Plot 1 is for the FFT of the audio
li2, = ax[1].plot(x, y)
ax[1].set_xlim(0,5000)
ax[1].set_ylim(0,1)
ax[1].set_title("Fast Fourier Transform")

# Show the plot, but without blocking updates
plt.pause(0.01)
plt.tight_layout()

class AudioHandler(object):
    def __init__(self):
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 1024 * 2
        self.p = None
        self.stream = None
        self.chunksRead = None
        self.predictor = Predictor()
    def start(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  output=False,
                                  stream_callback=self.callback,
                                  frames_per_buffer=self.CHUNK)

    def stop(self):
        self.stream.close()
        self.p.terminate()

    def callback(self, in_data, frame_count, time_info, flag):
        numpy_array = np.frombuffer(in_data, dtype=np.float32)
        # # librosa.feature.mfcc(numpy_array)
        self.chunksRead = in_data
        return None, pyaudio.paContinue

    def mainloop(self):

        while (self.stream.is_active()): # if using button you can set self.stream to 0 (self.stream = 0), otherwise you can use a stop condition
            if self.chunksRead != None:
                numpy_array = np.frombuffer(self.chunksRead, dtype=np.float32)
                data = numpy_array
                y = data
                Fs = self.RATE
                n = len(y)  # length of the signal
                k = np.arange(n)
                T = n / Fs
                frq = k / T  # two sides frequency range
                frq = frq[1: int(n / 2)]  # one side frequency range (sliced the frq in half)

                Y = np.fft.fft(y) / n  # fft computing and normalization
                Y = Y[1: int(n / 2)]
                Y2 = Y
                Y = Y*100
                Y = abs(Y)

                #get the peaks
                indexes = peakutils.indexes(Y, thres=0.01, min_dist=10)
                #print('first indexes:', indexes)
                fundamental = 0
                overtone_series = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                if len(indexes) >= 7 and Y[indexes[1]] > 0.05:
                    fundamental = indexes[1]
                    indexes = peakutils.indexes(Y, thres=0.01, min_dist=fundamental-5)
                    #print('new indexes:', indexes)

                    if len(indexes) > 7:
                        overtones = indexes[1:8]

                        if Y[indexes[1]] != 0:
                            overtone_series = [0]*700
                            temp = Y[overtones] / Y[indexes[1]]

                            for i in range(0,700,100):

                                for j in range(20):
                                    overtone_series[i+j] = temp[int(i/100)]

                li.set_xdata(np.arange(len(overtone_series)))
                li.set_ydata(overtone_series)
                ax[0].get_xaxis().set_visible(False)

                li2.set_xdata(frq)
                li2.set_ydata(abs(Y2)*10)

                plt.pause(0.01)

                rms = librosa.feature.rms(numpy_array)[0]

                for text in ax[0].texts:
                    text.set_visible(False)
                ax[0].text(300, 2, 'None', style='italic',
                           bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10})
                ax[0].texts[len(ax[0].texts)-1].set_visible(True)

                if min(rms) < 0.01:
                    continue

                print('peaks:',indexes[:8])
                print('fundamental_frequqncy: ~',fundamental*21.5)
                for text in ax[0].texts:
                    text.set_visible(False)
                ax[0].text(300, 2, self.predictor.getPrediction(numpy_array)[0].upper(), style='italic',
                           bbox={'facecolor': 'green', 'alpha': 1, 'pad': 10})
                ax[0].texts[len(ax[0].texts)-1].set_visible(True)

audio = AudioHandler()
audio.start()     # open the the stream
audio.mainloop()  # main operations with librosa
audio.stop()