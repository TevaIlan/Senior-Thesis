from __future__ import print_function
from orphics import maps,io,cosmology,catalogs,stats,mpi
from enlib import enmap
import numpy as np
import os,sys
from szar import counts,foregrounds
import matplotlib.pyplot as plt


import argparse
parser = argparse.ArgumentParser(description='Do a thing.')
parser.add_argument("cat", type=str,help='Positional arg.')
args = parser.parse_args()

io.dout_dir += args.cat+"_"

freqlist = ['90','150','217','353']
sfreqlist = ['90','150','217','353','545','857']
freqlist = [float(x) for x in sfreqlist]
lowfreqlist = ['90','150','217']
highfreqlist = ['353','545','857']

ccon = cosmology.defaultConstants

apmeans = []
aperrs = []


for freq in sfreqlist:
    aps,apwts = np.loadtxt("f"+freq+"_"+args.cat+"_apflux.txt",unpack=True)

    apmean = np.sum(aps*apwts)/np.sum(apwts)
    v1 = np.sum(apwts)
    v2 = np.sum(apwts**2.)
    aperr = np.sqrt(np.sum(apwts*(aps-apmean)**2.)/(v1-(v2/v1))) / np.sqrt(aps.size)
    
    
    print(freq, " S/N :",apmean/aperr)
    apmeans.append(apmean/1e6)
    aperrs.append(aperr/1e6)
  



def planck(nu_ghz,T):
    h = ccon['H_CGS']
    nu = 1e9*nu_ghz
    k = ccon['K_CGS']
    c = ccon['C']
    x = h*(nu)/(k*T)
    B = 2*h*nu**3.*(1./(np.exp(x)-1.))/c**2.
    return B


def dplanckT(nu_ghz):
    h = ccon['H_CGS']
    nu = 1e9*nu_ghz
    k = ccon['K_CGS']
    T = ccon['TCMB']
    c = ccon['C']
    x = h*(nu)/(k*T)
    dB = 2.*h**2.*nu**4.*(np.exp(x)/(np.exp(x)-1.)**2.)/c**2./k/T**2.
    return dB


def bdust(nu_ghz,z):
    nu0 = 353.
    beta = 1.78
    Tdust = 20.
    # print('Hi')
    # print(nu_ghz)
    # print(np.asarray(nu_ghz))
    # print(z)
    nu_ghz=np.asarray(nu_ghz)
    pref = (nu_ghz*(1+z)/nu0)**(beta)
    return pref * planck(nu_ghz*(1+z),Tdust) / dplanckT(nu_ghz)


def f_nu(nu):
    nu = np.asarray(nu)
    mu = ccon['H_CGS']*(1e9*nu)/(ccon['K_CGS']*ccon['TCMB'])
    ans = mu/np.tanh(mu/2.0) - 4.0
    return ans


zavg = 0.36315244

TCMB = 2.7255
def yflux(fghz,Yin):
    return TCMB*f_nu(fghz)*Yin
def dflux(fghz,Din):
    return bdust(fghz,zavg)*Din
def Sflux(fghz,Yin,Din):
    return yflux(fghz,Yin)+dflux(fghz,Din)#+dT

#from scipy.optimize import curve_fit
def gfunc(freq,Tcmb=2.7255e6):
  hplanck=6.626068e-34 #MKS
  kboltz=1.3806503e-23 #MKS
  X=hplanck*freq/(kboltz*Tcmb/1e6) 
  Xtwid=X*np.cosh(0.5*X)/np.sinh(0.5*X)
  Y0=Xtwid-4.0
  gfunc=Y0
  return gfunc

def gfuncrel(freq,Te,Tcmb=2.7255e6):
  """
  !-----------------------------------------------------------------------------
  ! From J.Colin Hill
  !
  ! Spectral function of the thermal Sunyaev-Zel'dovich effect.
  ! Relativistic corrections implemented using Nozawa et al. (2006)
  !   and Arnaud et al. (2005) for the T_e(M,z) relation.
  !
  !-----------------------------------------------------------------------------
  """

  
  hplanck=6.626068e-34 #MKS
  kboltz=1.3806503e-23 #MKS
    
  X=hplanck*freq/(kboltz*Tcmb/1e6) 
  Xtwid=X*np.cosh(0.5*X)/np.sinh(0.5*X)
  Stwid=X/np.sinh(0.5*X)
  Y0=Xtwid-4.0
  Y1=-10.0+23.5*Xtwid-8.4*Xtwid**2+0.7*Xtwid**3+ \
       Stwid**2*(-4.2+1.4*Xtwid)
  Y2=-7.5+127.875*Xtwid-173.6*Xtwid**2.0+65.8*Xtwid**3.0- \
       8.8*Xtwid**4.0+0.3666667*Xtwid**5.0+Stwid**2.0* \
       (-86.8+131.6*Xtwid-48.4*Xtwid**2.0+4.7666667*Xtwid**3.0)+ \
       Stwid**4.0*(-8.8+3.11666667*Xtwid)
  Y3=7.5+313.125*Xtwid-1419.6*Xtwid**2.0+1425.3*Xtwid**3.0- \
       531.257142857*Xtwid**4.0+86.1357142857*Xtwid**5.0- \
       6.09523809524*Xtwid**6.0+0.15238095238*Xtwid**7.0+ \
       Stwid**2.0*(-709.8+2850.6*Xtwid-2921.91428571*Xtwid**2.0+ \
       1119.76428571*Xtwid**3.0-173.714285714*Xtwid**4.0+ \
       9.14285714286*Xtwid**5.0)+Stwid**4.0*(-531.257142857+ \
       732.153571429*Xtwid-274.285714286*Xtwid**2.0+ \
       29.2571428571*Xtwid**3.0)+Stwid**6.0* \
       (-25.9047619048+9.44761904762*Xtwid)


  me = 9.109383e-31 # kg
  cspeed = 2.99792458e8 # m/s
  tTe = kboltz*Te/cspeed**2./me

  gfuncrel=Y0+Y1*tTe+Y2*tTe**2.0+Y3*tTe**3.0 
  return gfuncrel

freqs = np.arange(30,900,1.)

fnu = f_nu(freqs)
bnu = bdust(freqs,zavg)



def lnlike(param,nu,S,yerr): 
    #sys.exit()
    Y,D=param
    nu=np.asarray(nu)
    model=yflux(nu,Y)+dflux(nu,D)#+Y * gfuncrel(nu,Teff)
    yerr=np.asarray(yerr)
    inv_sigma2 = 1.0/(yerr**2)
    #print(param,-0.5*(np.sum((S-model)**2*inv_sigma2 - np.log(inv_sigma2))))
    return -0.5*(np.sum((S-model)**2*inv_sigma2- np.log(inv_sigma2)))

def lnprior(param):
    #sys.exit()
    Y,D=param
    if -1e-16 < Y < -1e-17 and  1e-18< D < 1e-17:# and -1e-16 < dT < 0:
        return 0.0
    return -np.inf

def lnprob(param, nu, S, yerr):
    lp = lnprior(param)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(param,nu,S,yerr)


ndim, nwalkers = 2, 20
guess=np.array([-7.96871772e-17,5.74833793e-18])
pos = [guess*(1+0.1*np.random.uniform(-1,1,size=ndim)) for i in range(nwalkers)]
#print(pos[0],pos[1])
#sys.exit()

# Y = 7e-16
# D = 3e-17

#dT = 0.
# ysyn = Sflux(freqlist,Y,D)
# yerrs = 0.3*np.abs(ysyn)
# ynoise = np.random.normal(0.,scale=yerrs)
# ysyn += ynoise


# param=6.9e-16,2.9e-17
# print(lnlike(param,freqlist, ysyn, yerrs))
# sys.exit()

#Uncomment this area later
import emcee
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(freqlist, apmeans, aperr))


sampler.run_mcmc(pos, 500000)
rawsamples=sampler.chain[:,:, :]
np.save("samples.npy",rawsamples)
samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
print(samples.shape)
Y_samples=rawsamples[:,:,0]

D_samples=rawsamples[:,:,1]
#dT_samples=rawsamples[:,:,2]
#print(Y_samples)
plt.hist(Y_samples.reshape(-1))
plt.title('Y')
plt.savefig('Y mcmc values.png')
plt.close()

plt.hist(D_samples.reshape(-1))
plt.title('D')
plt.savefig('D mcmc values.png')

# plt.hist(dT_samples)
# plt.title('dT')
# plt.savefig('dT mcmc values.png')

for i in range(nwalkers):
    plt.plot(Y_samples[i,:])
    #print(Y_samples[i,:])
ymin=np.min(Y_samples)
ymax=np.max(Y_samples)
plt.ylim(ymin,ymax)
plt.title('Y')
plt.savefig('Y mcmc chain.png')

for i in range(nwalkers):
    plt.plot(D_samples[i,:])
ymin=np.min(D_samples)
ymax=np.max(D_samples)
plt.ylim(ymin,ymax)
plt.title('D')
plt.savefig('D mcmc chain.png')
#Uncomment this area later

# plt.plot(dT_samples)
# ymin=np.min(dT_samples)
# ymax=np.max(dT_samples)
# plt.ylim(ymin,ymax)
# plt.title('dT')
# plt.savefig('dT mcmc chain.png')

# plt.contour(Y_samples,D_samples)
# plt.savefig('contour.png')

# plt.hist(Teff_samples)
# plt.title('Teff')
# plt.savefig('Teff mcmc values.png')

#samples[:, 1] = np.exp(samples[:, 1])
Y_mcmc, D_mcmc= map(lambda v: (v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))
print(Y_mcmc, D_mcmc)
print(np.mean(Y_samples))
print(np.mean(D_samples))
#print('Hi')



# Y = 4.5e-17
# D = 9e-18
Y = -7.96871772e-17
D = 5.74833793e-18


dT = 0.
ysyn = Sflux(freqlist,Y,D)
yerrs = 0.25*np.abs(ysyn)
#yerrs = 0.01*np.abs(ysyn)
ynoise = np.random.normal(0.,scale=yerrs)
ysyn += ynoise

# param=6.9e-16,2.9e-17
# print(lnlike(param,freqlist, ysyn, yerrs))
# sys.exit()

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(freqlist, ysyn, yerrs))
sampler.run_mcmc(pos, 50000)
sim_samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

plt.close()
plt.errorbar(freqlist,np.abs(apmeans),aperrs,label='data',marker='o')
plt.errorbar(freqlist,np.abs(ysyn),yerrs,label='simulated',marker='o')
plt.legend()
plt.yscale('log')
plt.savefig('simulated and data.png')
print(aperrs)
print(yerrs)

import corner
fig = corner.corner(samples, labels=["$Y$", "$D$"])#,
                     #truths=[Y, D])
fig1 = corner.corner(sim_samples, labels=["$Y$", "$D$"],
                     truths=[Y, D])
fig.savefig("triangle500000.png")
fig1.savefig("simulatedtriangle.png")
