#========================================================================
#                                  IMPORTS
#------------------------------------------------------------------------
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import sys
pi = np.pi
#========================================================================
#                                  INPUTS
#------------------------------------------------------------------------
# Sinais analisados
#
periodo_1 = 10
n_pontos = 50
n_total = n_pontos
n = np.arange(n_total)
f_1 = np.cos(2*pi*n/periodo_1)
#
periodo_2 = 10
n_pontos = 50
n_total = n_pontos
n = np.arange(n_total)
f_2 = np.cos(2*pi*n/periodo_2)
#========================================================================
#                            FUNCOES AUXILIARES
#------------------------------------------------------------------------
def psd(signal, sr=1):
    '''
    numpy.fft.fft:
    When the input a is a time-domain signal and A = fft(a): 
    . np.abs(A) is its amplitude spectrum; 
    . np.abs(A)**2 is its power spectrum; 
    . np.angle(A) is the phase spectrum.
    '''
    f = signal
    N = 2048
    ft = fft.fft(f, N)
    ft_shifted = fft.fftshift(ft)
    aft = np.abs(ft)**2
    aft_shifted = abs(ft_shifted)**2
    #
    freq = np.fft.fftfreq(N, d=sr)
    freqs = np.concatenate([freq[int(len(freq)/2):],[0]])
    freqs = np.concatenate([freqs, freq[1:int(len(freq)/2)]])
    #
    return freqs, aft_shifted
#
def rect(signal, width=10):
    center = int(n_pontos/2)
    x=np.arange(len(signal))
    signal_windowed = np.where(abs(center-x)<=width, signal, 0)
    return signal_windowed
#========================================================================
#                                 PLOTS
#------------------------------------------------------------------------
#------------------------------------------------------------------------ Plot 1
fig1, ax = plt.subplots(4, 2, figsize=(9,7))
#------------------------------------ 1 linha
n_analitico=1000
nn = np.arange(n_analitico)
f= np.cos(2*pi*nn/periodo_1)
xf=np.arange(-n_analitico/2,+n_analitico/2)
s1 = f_1
xs1 = np.arange(len(s1))
#-------------- sinal
ax[0,0].plot(xf, f, 'k')
ax[0,0].set_ylabel('Infinite\n signal ', rotation=0, labelpad=25.)
ax[0,0].set_title('Signal')
#-------------- psd
ax[0,1].vlines(1/10, 0, 1.5)
ax[0,1].vlines(-1/10, 0, 1.5)
ax[0,1].hlines(0, -.5, .5)
ax[0,1].set_title('Power Spectrum')
#------------------------------------ 2 linha
s2 = rect(f_1, 25)
xs2 = np.arange(len(s2))
#-------------- sinal
ax[1,0].plot(xs2, s2, 'k')
ax[1,0].set_ylabel('window\n width: 50', rotation=0, labelpad=25.)
#-------------- psd
x2, y2 = psd(s2)
ax[1,1].plot(x2, y2, 'k')
#------------------------------------ 3 linha
s3 = rect(f_1, 10)
xs3 = np.arange(len(s3))
#-------------- sinal
ax[2,0].plot(xs3, s3, 'k')
ax[2,0].set_ylabel('window\n width: 20', rotation=0, labelpad=25.)
#-------------- psd
x3, y3 = psd(s3)
ax[2,1].plot(x3, y3, 'k')
#------------------------------------ 4 linha
s4 = rect(f_1, 5)
xs4 = np.arange(len(s4))
#-------------- sinal
ax[3,0].plot(xs4, s4, 'k')
ax[3,0].set_ylabel('window\n width: 10', rotation=0, labelpad=25.)
#-------------- psd
x4, y4 = psd(s4)
ax[3,1].plot(x4, y4, 'k')
#------------------------------------------------------------------------
for i in range(4):
    ax[i,0].set_ylim(-1.2, 1.2)
    ax[i,1].set_yticks([])
    ax[i,0].set_xlim(-3,n_pontos+3)
    ax[i,1].set_xlim(-.5,.5)
    ax[i,1].set_xticks([-.4, -.3, -.2, -.1, 0., .1, .2, .3, .4])
    ax[i,1].set_xticklabels(['-0.4', '', '-0.2', '', '0.0', '', '0.2', '', '0.4'])
#------------------------------------------------------------------------
#fig1.suptitle('Effects of Windowing')
fig1.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.2, hspace=0.3)
fig1.savefig('fig1.jpg', dpi=400, bbox_inches='tight')
plt.show()
#------------------------------------------------------------------------ Plot 2
fig2, ax = plt.subplots(4, 2, figsize=(9,7))
#------------------------------------ 1 linha
sr = 1
s1 = f_2[0::sr]
xs1 = np.arange(0,len(f_2),sr)
#-------------- sinal
ax[0,0].plot(xs1, s1, color='darkgray', linestyle=(0, (3, 5, 1, 5)))
ax[0,0].plot(xs1, s1, 'k.')
ax[0,0].set_ylabel('Sampling\n rate: '+str(sr), rotation=0, labelpad=25.)
ax[0,0].set_title('Signal')
#-------------- psd
x1, y1 = psd(s1)
ax[0,1].plot(x1, y1, 'k')
ax[0,1].set_title('Power Spectrum')
#------------------------------------ 2 linha
sr = 3
s2 = s1[0::sr]
xs2 = xs1[0::sr]
#-------------- sinal
n2 = np.cos(2*pi*n/30.)
ax[1,0].plot(xs1, n2, color='darkgray', linestyle=(0, (3, 5, 1, 5)))
ax[1,0].plot(xs2, s2, 'k.')
ax[1,0].set_ylabel('Sampling\n rate: '+str(sr), rotation=0, labelpad=25.)
#-------------- psd
x2, y2 = psd(s2)
ax[1,1].plot(x2, y2, 'k')
#------------------------------------ 3 linha
sr = 4
s3 = s1[0::sr]
xs3 = xs1[0::sr]
#-------------- sinal
n3 = np.cos(2*pi*n/40.)
ax[2,0].plot(xs1, n3, color='darkgray', linestyle=(0, (3, 5, 1, 5)))
ax[2,0].plot(xs3, s3, 'k.')
ax[2,0].set_ylabel('Sampling\n rate: '+str(sr), rotation=0, labelpad=25.)
#-------------- psd
x3, y3 = psd(s3)
ax[2,1].plot(x3, y3, 'k')
#------------------------------------ 4 linha
sr = 5
s4 = s1[0::sr]
xs4 = xs1[0::sr]
#-------------- sinal
n4 = np.cos(2*pi*n/50.)
ax[3,0].plot(xs1, n4, color='darkgray', linestyle=(0, (3, 5, 1, 5)))
ax[3,0].plot(xs4, s4, 'k.')
ax[3,0].set_ylabel('Sampling\n rate: '+str(sr), rotation=0, labelpad=25.)
#-------------- psd
x4, y4 = psd(s4)
ax[3,1].plot(x4, y4, 'k')
#------------------------------------------------------------------------
for i in range(4):
    ax[i,0].set_ylim(-1.2, 1.2)
    ax[i,1].set_yticks([])
    ax[i,0].set_xlim(-3,n_pontos+3)
    ax[i,1].set_xlim(-.5,.5)
    ax[i,1].set_xticks([-.4, -.3, -.2, -.1, 0., .1, .2, .3, .4])
    ax[i,1].set_xticklabels(['-0.4', '', '-0.2', '', '0.0', '', '0.2', '', '0.4'])
    #ax[i,1].set_xlim(-.4,.4)
#------------------------------------------------------------------------
#fig2.suptitle('Effects of Sampling')
fig2.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.2, hspace=.3)
fig2.savefig('fig2.jpg', dpi=400, bbox_inches='tight')
plt.show()
#------------------------------------------------------------------------ Plot 3
fig3, ax = plt.subplots(4, 2, figsize=(9,7))
#------------------------------------ 1 linha
g1 = [0+3,1+3,2+3,3+3,4+3, 45-4,46-4,47-4,48-4,49-4]
xs1=[]
for i in np.arange(0,len(f_2)-1):
    if i not in g1:
        xs1.append(i)
s1 = f_2[xs1]
#-------------- sinal
ax[0,0].plot(xs1, s1, 'k.')
ax[0,0].set_title('Signal')
ax[0,0].set_ylabel('Number\n of gaps: 2', rotation=0, labelpad=35.)
#-------------- psd
x1, y1 = psd(s1)
ax[0,1].plot(x1, y1, 'k')
ax[0,1].set_title('Power Spectrum')
#------------------------------------ 2 linha
g2 = [10-4,11-4,12-4,13-4,14-4, 20,21,22,23,24, 35-3,36-3,37-3,38-3,39-3, 45,46,47,48,49]
xs2=[]
for i in np.arange(0,len(f_2)-1):
    if i not in g2:
        xs2.append(i)
s2 = f_2[xs2]
#-------------- sinal
ax[1,0].plot(xs2, s2, 'k.')
ax[1,0].set_ylabel('Number\n of gaps: 4', rotation=0, labelpad=35.)
#-------------- psd
x2, y2 = psd(s2)
ax[1,1].plot(x2, y2, 'k')
#------------------------------------ 3 linha
xs3 = [0]
while len(xs3)<len(f_2)/2:
    i = np.random.randint(len(f_2))
    if i not in xs3:
        xs3.append(i)
s3 = []
for i in range(len(f_2)):
    if i in xs3:
        s3.append(f_2[i])
#-------------- sinal
ax[2,0].plot(xs3, s3, 'k.')
ax[2,0].set_ylabel('Random gaps: \n'+str(round(100*len(xs3)/len(f_2),2))+'% of data', rotation=0, labelpad=35.)
#-------------- psd
x3, y3 = psd(s3)
ax[2,1].plot(x3, y3, 'k')
#------------------------------------ 4 linha
xs4 = [0]
while len(xs4)<len(f_2)/4:
    i = np.random.randint(len(f_2))
    if i not in xs4:
        xs4.append(i)
s4 = []
for i in range(len(f_2)):
    if i in xs4:
        s4.append(f_2[i])
#-------------- sinal
ax[3,0].plot(xs4, s4, 'k.')
ax[3,0].set_ylabel('Random gaps: \n'+str(round(100*len(xs4)/len(f_2),1))+'% of data', rotation=0, labelpad=35.)
#-------------- psd
x4, y4 = psd(s4)
ax[3,1].plot(x4, y4, 'k')
#------------------------------------------------------------------------
for i in range(4):
    ax[i,0].set_ylim(-1.2, 1.2)
    ax[i,1].set_yticks([])
    ax[i,0].set_xlim(-3,n_pontos+3)
    ax[i,1].set_xlim(-.5,.5)
    ax[i,1].set_xticks([-.4, -.3, -.2, -.1, 0., .1, .2, .3, .4])
    ax[i,1].set_xticklabels(['-0.4', '', '-0.2', '', '0.0', '', '0.2', '', '0.4'])
#------------------------------------------------------------------------
fig3.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.2, hspace=.3)
fig3.savefig('fig3.jpg', dpi=400, bbox_inches='tight')
plt.show()
