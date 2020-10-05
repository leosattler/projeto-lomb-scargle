#========================================================================
#                                 IMPORTS
#------------------------------------------------------------------------
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
#import scipy.signal as signal
from astropy.timeseries import LombScargle
#========================================================================
#                           FUNCOES AUXILIARES
#------------------------------------------------------------------------
def tratamento_jump(data, times_media=5):
    m = np.mean(data)
    data[np.where(data>times_media*m)] = m
#========================================================================
#                                  SINAIS
#------------------------------------------------------------------------
years = np.arange(1964, 2020)
#-------------------------------------
'''
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
f3_file = np.genfromtxt('../../dados/yearlyaveraged_data.txt')
f3_year = f3_file[1:, 0]
f3_doy = f3_file[1:, 1]
f3_data = f3_file[1:, 3]
f3 = np.array(f3_data)[:-1]
f1=np.array(f3)
'''
#========================================================================
#                        CRIANDO GAPS NOS DADOS
#------------------------------------------------------------------------
f = np.array(f1)
#-----------------------------------
# model = 1, 2 or 3
# 1 - pg (% of N = Ng/N) gaps of size 1, randomly placed (artigo)
# 2 - Ng gaps of size Dg, randomly placed (professora)
# 3 - Ng gaps of size Dg, regularly placed with a period of T (me) 
#-----------------------------------
#model = 1
#pg = .98
#-----------------------------------
model = 2
Ng = 2
Dg = int(len(f)/10)
#-----------------------------------
#model = 3
#Ng = 5
#Dg = int(len(f)/10)
#-----------------------------------
x = np.arange(len(f))
#-----------------------------------
if model == 1:
    erased = []
    while len(erased)/len(f) <= pg:
        #print(len(erased)/len(f), pg)
        to_erase = np.random.randint(len(f))
        if to_erase not in erased:
            erased.append(to_erase)
#-----------------------------------
to_erase_center=0
if model == 2:
    erased = []
    Ng_k = 0
    kk=0
    while Ng_k < Ng:
        print(Ng_k, Ng, to_erase_center)
        to_erase_center = np.random.randint(Dg/2+1, len(f)-Dg/2-1)
        if (to_erase_center - np.ceil(Dg/2) ) not in erased and \
           (to_erase_center + np.ceil(Dg/2) ) not in erased:
            for i in np.arange(to_erase_center - np.ceil(Dg/2), to_erase_center + np.ceil(Dg/2), dtype=int):
                erased.append(int(i))
            #
            Ng_k = Ng_k + 1
#-----------------------------------
if model == 3:
    erased = []
    centers = np.linspace(0, len(f), Ng+2)
    for i in centers[1:-1]:
        for j in np.arange(i - Dg/2, i + Dg/2):
            erased.append(int(j))
#-----------------------------------
fg = []
xg = []
for i in range(len(f)):
    if i not in erased:
        print(i, len(f))
        fg.append(f[i])
        xg.append(i)
fg = np.array(fg)
xg = np.array(xg)
#========================================================================
#                        LOMB-SCARGLE PERIODOGRAM
#------------------------------------------------------------------------
N1 = len(f1)
freq = np.fft.fftfreq(N1, d=1/sr)
frequency = np.linspace(freq[1:int(N1/2)-1][0], freq[1:int(N1/2)-1][-1], 1000)
#------------------------------------
# Astropy
#p_ls = LombScargle(xg, fg).power(freqs)
#freqs, p_ls = LombScargle(xg, fg).autopower(minimum_frequency=0,\
#                                         maximum_frequency=0.5,\
#                                         normalization='psd',\
#                                         samples_per_peak=20)
ny=2.
freqs, p_ls = LombScargle(xg, fg).autopower(nyquist_factor=ny)
#========================================================================
#                                 PLOT
#------------------------------------------------------------------------
fig = plt.figure(figsize=(9,7))
#------------------------------------
plt.subplot(3,1,1)
if model==2:
    plt.title('Data with '+str(Ng)+' gaps - ' + str(round(100*(len(fg)/len(f1)),2))+'% of data')
else:
    plt.title(str(round(100*(1-pg),2))+'% of data')
#plt.plot(freq[1:int(N1/2)-1], aft1[1:int(N1/2)-1], 'r-')
plt.plot(xg, fg, 'k.')
plt.xlim(-.045*max(x), max(x)+ .045*max(x))
#plt.xscale('log')
#------------------------------------
plt.subplot(3,1,2)
plt.title('Lomb-Scargle periodogram')
plt.plot(sr*freqs, p_ls,'k-')
#plt.xlim(0.01,100)
best_frequency = freqs[np.argmax(p_ls)]
plt.vlines(sr*best_frequency, 0, 1.1*np.max(p_ls), label = 'freq. = '+str(round(sr*best_frequency,4))+','+'\n' + 'period = '+str(round(1/(sr*best_frequency),4)), linestyle='dashed')
plt.legend(loc=0)
plt.xscale('log')
#------------------------------------
plt.subplot(3,1,3)
t_fit = x#np.linspace(0, 1)
ls = LombScargle(xg, fg)
y_fit = ls.model(t_fit, best_frequency)
#
mean_res = np.mean(np.sqrt((y_fit - f1)**2))
#
plt.plot(x, f1, 'k.', label='data')
plt.plot(t_fit,y_fit,'r-', label='fitted, res. = ' + str(round(mean_res,1)))
plt.xlim(-.045*max(x), max(x)+ .045*max(x))
plt.legend(loc=0)
#
#------------------------------------
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.4)
if model == 1:
    fig.savefig('periodograms_ny'+str(ny)+'_model'+str(model)+'_pg'+str(pg)+'.jpg', dpi = 400, bbox_inches='tight')
else:
    fig.savefig('periodograms_ny'+str(ny)+'_model'+str(model)+'_Ng'+str(Ng)+'.jpg', dpi = 400, bbox_inches='tight')
plt.show()
