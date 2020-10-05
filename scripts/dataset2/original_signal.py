#========================================================================
#                                 IMPORTS
#------------------------------------------------------------------------
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
#========================================================================
#                           FUNCOES AUXILIARES
#------------------------------------------------------------------------
def tratamento_jump(data, times_media=5):
    m = np.mean(data)
    data[np.where(data>times_media*m)] = m
#
def psd(signal, sr=1, N=2048):
    '''
    numpy.fft.fft:
    When the input a is a time-domain signal and A = fft(a): 
    . np.abs(A) is its amplitude spectrum; 
    . np.abs(A)**2 is its power spectrum; 
    . np.angle(A) is the phase spectrum.
    '''
    f = signal
    ft = fft.fft(f, N)/N
    ft_shifted = fft.fftshift(ft)
    aft = np.abs(ft)**2
    aft_shifted = abs(ft_shifted)**2
    #
    freq = np.fft.fftfreq(N, d=1/sr)
    freqs = np.concatenate([freq[int(len(freq)/2):],[0]])
    freqs = np.concatenate([freqs, freq[1:int(len(freq)/2)]])
    #
    return freqs, aft_shifted
#========================================================================
#                                  SINAIS
#------------------------------------------------------------------------
years = np.arange(1964, 2020)
#-------------------------------------
'''
sr = 365
f1_file=np.genfromtxt('../../dados/dailyaveraged_data.txt')
f1_year = f1_file[1:, 0]
f1_doy = f1_file[1:, 1]
f1_data = f1_file[1:, 3]
tratamento_jump(f1_data, 5)
f1 = []
k=0
for y in years:
    c = 0
    for i in np.arange(0, len(f1_data)):
        if int(f1_year[i]) == int(y) and c<=365:
            d = f1_data[i]
            f1.append(d)
            c=c+1
        if c>=365:
            break
f1 = np.array(f1)
'''
#-------------------------------------
sr = 12
f2_file = np.genfromtxt('../../dados/27dayaveraged_data.txt')
f2_year = f2_file[1:, 0]
f2_doy = f2_file[1:, 1]
f2_data = f2_file[1:, 3]
f2 = []
k=0
for y in years:
    c=0
    for i in np.arange(0, len(f2_data)):
        if int(f2_year[i]) == int(y) and c<=12:
            d = f2_data[i]
            f2.append(d)
            c=c+1
        if c>=12:
            break
f1=np.array(f2)
#-------------------------------------
'''
sr = 1
f3_file = np.genfromtxt('../../dados/yearlyaveraged_data.txt')
f3_year = f3_file[1:, 0]
f3_doy = f3_file[1:, 1]
f3_data = f3_file[1:, 3]
f3 = np.array(f3_data)[:-1]
f1=np.array(f3)
'''
#========================================================================
#                                  PLOT
#------------------------------------------------------------------------
f = np.array(f1)
x=np.arange(len(f))
#------------------------------------
fig = plt.figure(figsize=(7,5))
#------------------------------------
plt.subplot(2,1,1)
plt.title('27-days averaged signal (uniformly sampled)')
#plt.plot(freq[1:int(N1/2)-1], aft1[1:int(N1/2)-1], 'r-')
plt.plot(x, f, 'k.')
plt.xlim(-.045*max(x), max(x)+ .045*max(x))
#plt.xscale('log')
#------------------------------------
plt.subplot(2,1,2)
plt.title('Power Spectrum via Fourier')
freqs, aft_shifted = psd(f, sr, 2**9)
plt.plot(freqs, aft_shifted,'ko-')
freqs_real = freqs[int(len(aft_shifted)/2)+2:]
best_frequency = freqs_real[np.argmax(aft_shifted[int(len(aft_shifted)/2)+2:])]
plt.vlines(best_frequency, 0, 1.1*np.max(aft_shifted), label = 'freq. = '+str(round(best_frequency,4))+','+'\n' + 'period = '+str(round(1/(best_frequency),4)), linestyle='dashed')
plt.ylim(0,1000)
plt.legend(loc=0)
plt.xscale('log')
#------------------------------------
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.4)
fig.savefig('original_'+str(sr)+'.jpg', dpi = 400, bbox_inches='tight')
plt.show()
