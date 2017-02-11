import scipy
import numpy as np
import pylab
from matplotlib.font_manager import FontProperties
import scipy.io.wavfile
from scipy.misc import imsave

from scipy import ceil, complex64, float64, hamming, zeros
from scipy.fftpack import fft# , ifft
from scipy import ifft # こっちじゃないとエラー出るときあった気がする

from matplotlib import pylab as pl

from PIL import Image

def stft(x, win, step):
    l = len(x) # 入力信号の長さ
    N = len(win) # 窓幅、つまり切り出す幅
    M = int(ceil(float(l - N + step) / step)) # スペクトログラムの時間フレーム数
    
    new_x = zeros(N + ((M - 1) * step), dtype = float64)
    new_x[: l] = x # 信号をいい感じの長さにする
    
    X = zeros([M, N], dtype = complex64) # スペクトログラムの初期化(複素数型)
    for m in range(M):
        start = step * m
        X[m, :] = fft(new_x[start : start + N] * win)
    return X

def istft(X, win, step):
    M, N = X.shape
    assert (len(win) == N), "FFT length and window length are different."

    l = (M - 1) * step + N
    x = zeros(l, dtype = float64)
    wsum = zeros(l, dtype = float64)
    for m in range(M):
        start = step * m
        ### 滑らかな接続
        x[start : start + N] = x[start : start + N] + ifft(X[m, :]).real * win
        wsum[start : start + N] += win ** 2 
    pos = (wsum != 0)
    x_pre = x.copy()
    ### 窓分のスケール合わせ
    x[pos] /= wsum[pos]
    return x

fs, data = scipy.io.wavfile.read("test.wav")
fftLen = 2048 # とりあえず
win = scipy.hamming(fftLen) # ハミング窓
step = fftLen / 4
spectrogram = stft(data, win, step)

spectrogram_real = np.real(spectrogram)
spectrogram_real = spectrogram_real * (128 / spectrogram_real.min())
spectrogram_real = spectrogram_real + 128

spectrogram_imag = np.imag(spectrogram)
spectrogram_imag = spectrogram_imag * (128 / spectrogram_imag.min())
spectrogram_imag = spectrogram_imag + 128

im1 = Image.fromarray(np.uint8(spectrogram_real))
im1.save("test_r.png")

im2 = Image.fromarray(np.uint8(spectrogram_imag))
im2.save("test_i.png")