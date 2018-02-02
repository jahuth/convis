import matplotlib.pyplot as plt
import numpy as np
import convis
# random amplitudes and phases for a range of frequencies
signal = np.sum([np.random.randn()*np.sin(np.linspace(0,2.0,5000)*(freq)
                 + np.random.rand()*2.0*np.pi)
                 for freq in np.logspace(-2,7,136)],0)
f1 = convis.filters.simple.TemporalLowPassFilterRecursive()
f1.tau.data[0] = 0.005
f2 = convis.filters.simple.TemporalHighPassFilterRecursive()
f2.tau.data[0] = 0.005
f2.k.data[0] = 1.0
o1 = f1(signal[None,None,:,None,None]).data.numpy().mean((0,1,3,4)).flatten()
o2 = f2(signal[None,None,:,None,None]).data.numpy().mean((0,1,3,4)).flatten()
plt.plot(signal,label='Signal')
plt.plot(o2,label='High Pass Filter')
plt.plot(o1,label='Low Pass Filter')
signal_f = np.fft.fft(signal)
o1_f = np.fft.fft(o1)
o2_f = np.fft.fft(o2)
plt.legend()
plt.figure()
plt.plot(0,0)
plt.plot(np.abs(o2_f)[:2500]/np.abs(signal_f)[:2500],label='High Pass Filter')
plt.plot(np.abs(o1_f)[:2500]/np.abs(signal_f)[:2500],label='Low Pass Filter')
plt.xlabel('Frequency')
plt.ylabel('Gain')
plt.title('Transfer Functions of Filters')
plt.gca().set_xscale('log')
plt.ylim(-0.1,1.25)
plt.legend()