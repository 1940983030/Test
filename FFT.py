import numpy as np
import numpy.fft as nf
import scipy.io.wavfile as wf
import matplotlib.pyplot as plt

# 读取音频文件
sample_rate, noised_sigs = wf.read('test-1k.wav')
print(sample_rate)  # sample_rate：采样率44100
print(noised_sigs.shape)  # noised_sigs:存储音频中每个采样点的采样位移(220500,)
times = np.arange(noised_sigs.size) / sample_rate

# 傅里叶变换后，绘制频域图像
freqs = nf.fftfreq(times.size, times[1] - times[0])
complex_array = nf.fft(noised_sigs)
pows = np.abs(complex_array)

# 寻找能量最大的频率值
fund_freq = freqs[pows.argmax()]
# where函数寻找那些需要抹掉的复数的索引
noised_indices = np.where(freqs != fund_freq)
# 复制一个复数数组的副本，避免污染原始数据
filter_complex_array = complex_array.copy()
filter_complex_array[noised_indices] = 0
filter_pows = np.abs(filter_complex_array)

plt.plot(freqs[freqs >= 0], filter_pows[freqs >= 0], c='dodgerblue', label='Filter')
plt.xlabel('Frequency', fontsize=12)
plt.ylabel('Power', fontsize=12)
# plt.xlim(0, 8000, 10)
# plt.ylim(0, 100, 1)
plt.tick_params(labelsize=10)
plt.grid(linestyle=':')
plt.legend()
plt.show()
