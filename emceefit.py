from __future__ import print_function
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

#freqlist = ['90','150','217','353']
freqlist = ['90','150','217','353','545','857']
lowfreqlist = ['90','150','217']
highfreqlist = ['353','545','857']

ccon = cosmology.defaultConstants

apmeans = []
aperrs = []
# apmeans_lessthan353=[]
# aperrs_lessthan353=[]
# apmeans_greaterthan217=[]
# aperrs_greaterthan217=[]

#N=0

for freq in freqlist:
    aps,apwts = np.loadtxt("f"+freq+"_"+args.cat+"_apflux.txt",unpack=True)

    apmean = np.sum(aps*apwts)/np.sum(apwts)
    v1 = np.sum(apwts)
    v2 = np.sum(apwts**2.)
    aperr = np.sqrt(np.sum(apwts*(aps-apmean)**2.)/(v1-(v2/v1))) / np.sqrt(aps.size)
    
    # apmean = np.average(aps, weights=apwts)
    # aperr = np.sqrt(np.average((aps-apmean)**2, weights=apwts))/np.sqrt(aps.size)
    
    # apmean = aps.mean()
    # aperr = np.std(aps)/np.sqrt(aps.size)
    print(freq, " S/N :",apmean/aperr)
    apmeans.append(apmean/1e6)
    aperrs.append(aperr/1e6)
    # if N<3:
    #     apmeans_lessthan353.append(apmean/1e6)
    #     aperrs_lessthan353.append(aperr/1e6)
    # if N>2:
    #     apmeans_greaterthan217.append(apmean/1e6)
    #     aperrs_greaterthan217.append(aperr/1e6)
    # N=N+1



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
def Sflux(fghz,Yin,Din,dT):
    # if Yin<0: return np.nan
    # if Din<0: return np.nan
    return yflux(fghz,Yin)+dflux(fghz,Din)+dT

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

# Y = 7e-16
# D = 3e-17

# Y = 4e-17
# D = 2e-18

# fs = [float(f) for f in freqlist]
# pl = io.Plotter(xlabel='$\\nu$ (GHz)',ylabel='F (K*arcmin$^2$)',yscale='log')
# pl.add_err(fs,np.abs(apmeans),yerr=aperrs,marker="o",markersize=8,elinewidth=3)
# pl.add(freqs,np.abs(TCMB*Y*fnu),label='$T_{cmb} Y f_{nu}$')
# pl.add(freqs,np.abs(D*bnu),label='$D b_{nu}$')
# pl.add(freqs,np.abs(D*bnu+TCMB*Y*fnu),ls="--", label='$D b_{nu}+ T_{cmb} Y f_{nu}$')
# pl.hline()
# pl.legend()
# pl.done(io.dout_dir+"apfluxes.png")



# popt,pcov = curve_fit(Sflux,fs,apmeans,sigma=aperrs,p0=[Y,D,0.])

# print(popt)
# print(pcov)


# Yfit = popt[0]
# Dfit = popt[1]
# tfit = popt[2]

# sfit = Sflux(freqs,Yfit,Dfit,tfit)
# yfit = yflux(freqs,Yfit)
# dfit = dflux(freqs,Dfit)




# pl = io.Plotter(xlabel='$\\nu$ (GHz)',ylabel='F (K*arcmin$^2$)',yscale='log')
# pl.add_err(fs,np.abs(apmeans),yerr=aperrs,marker="o",markersize=8,elinewidth=3)#, label='Error')
# # pl.add(freqs,TCMB*Y*fnu)
# # pl.add(freqs,D*bnu)
# pl.add(freqs,np.abs(sfit),ls="-", label='sfit' )
# pl.add(freqs,np.abs(yfit),ls="--", label='yfit')
# pl.add(freqs,np.abs(dfit),ls="--",label='dfit')
# pl.hline()
# pl.hline(y=tfit,ls="-",alpha=0.2,label='kSZ')
# pl.legend()
# pl.done(io.dout_dir+"apfluxes_fitlog.png")


# pl = io.Plotter(xlabel='$\\nu$ (GHz)',ylabel='F (K*arcmin$^2$)')
# pl.add_err(fs,(apmeans),yerr=aperrs,marker="o",markersize=8,elinewidth=3)# label='Error')
# # pl.add(freqs,TCMB*Y*fnu)
# # pl.add(freqs,D*bnu)
# pl.add(freqs,(sfit),ls="-",label='sfit')
# pl.add(freqs,(yfit),ls="--",label='yfit')
# pl.add(freqs,(dfit),ls="--",label='dfit')
# pl.hline()
# pl.hline(y=tfit,ls="-",alpha=0.2,label='kSZ')
# pl.legend()
# pl.done(io.dout_dir+"apfluxes_fit.png")

# x = np.array([90,150,217,353,545,857])
# aperrs=np.array(aperrs)
# apmeans=np.array(apmeans)
# bfit,bcov,chisquare,pte = stats.fit_linear_model(x,apmeans,ycov=np.diag(aperrs**2.), funcs=[lambda x: 1.,lambda x: yflux(x,1.),lambda x: dflux(x,1.)])
# dTfit,Yfit,Dfit = bfit
# edTfit,eYfit,eDfit = np.sqrt(np.diagonal(bcov))

def lnlike(param,nu,S,yerr):
    Y,D,dT,Teff=param
    model=TCMB*f_nu(nu)*Y+pref * planck(nu*(1+z),Tdust) / dplanckT(nu)*D+dT+Y * gfuncrel(nu,Teff)
    inv_sigma2 = 1.0/(yerr**2 + model**2)
    return -0.5*(np.sum((S-model)**2*inv_sigma2 - np.log(inv_sigma2)))

def lnprior(param):
    Y,D,dT,Teff=param
    if -1e-15 < Y < -1e-16 and  1e-17< D < 1e-16 and -1e-16 < dT < 0 and -1e-15 < Teff < 0:
        return 0.0
    return -np.inf

def lnprob(param, nu, S, yerr):
    lp = lnprior(param)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(param,nu,S,yerr)

freqarray=np.array(freqlist)
# apmeans=np.array(apmeans)
# aperrs=np.array(aperrs)
# import scipy.optimize as op
# nll = lambda *args: -lnlike(*args)
# result = op.minimize(nll, [Y_true, D_true, dT_true,Teff_true], args=(freqarray, apmeans, aperrs))
ndim, nwalkers = 4, 100
p0 = [np.random.rand(ndim) for i in xrange(nwalkers)]
#pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
import emcee
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(freqarray, apmeans, aperrs))
pos, prob, state = sampler.run_mcmc(p0, 200)
sampler.reset()
sampler.run_mcmc(pos, 500)
samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
print(samples.shape)
Y_samples=samples[:,0]
D_samples=samples[:,0]
dT_samples=samples[:,0]
Teff_samples=samples[:,0]
plt.hist(Y_samples)
plt.title('Y')
plt.savefig('Y mcmc values.png')

plt.hist(D_samples)
plt.title('D')
plt.savefig('D mcmc values.png')

plt.hist(dT_samples)
plt.title('dT')
plt.savefig('dT mcmc values.png')

plt.hist(Teff_samples)
plt.title('Teff')
plt.savefig('Teff mcmc values.png')

samples[:, 2] = np.exp(samples[:, 2])
Y_mcmc, D_mcmc, dT_mcmc, Teff_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))
print(Y_mcmc, D_mcmc, dT_mcmc, Teff_mcmc)
#print(dT,Y,D)
# print(dTfit,Yfit,Dfit)
# print(dTfit/edTfit,Yfit/eYfit,Dfit/eDfit)
# print(chisquare,pte)
# print(bcov)
# bcor=stats.cov2corr(bcov)
# print(bcor)
# print(freqs)
# print(Yfit[0])
# print(Dfit[0])
# print(dTfit[0])
# print(Sflux(freqs,Yfit[0],Dfit[0],dTfit[0]))
# print(np.abs(Sflux(freqs,Yfit[0],Dfit[0],dTfit[0])))

# e


# print(apmeans)
# print(apmeans_lessthan353)
# fs = [float(f) for f in lowfreqlist]
# #freqs = np.arange(30,250,1.)
# pl = io.Plotter(xlabel='$\\nu$ (GHz)',ylabel='F (K*arcmin$^2$)')
# pl.add_err(fs,(apmeans_lessthan353),yerr=aperrs_lessthan353,marker="o",markersize=8,elinewidth=3)# label='Error')
# # pl.add(freqs,TCMB*Y*fnu)
# # pl.add(freqs,D*bnu)
# pl.add(freqs,(sfit),ls="-",label='sfit')
# pl.add(freqs,(yfit),ls="--",label='yfit')
# pl.add(freqs,(dfit),ls="--",label='dfit')
# pl.legend()
# pl.hline()
# # pl.hline(y=tfit,ls="-",alpha=0.2)
# pl.done(io.dout_dir+"apfluxes_fit_lessthan353.png")

# D = 1e-7
# Y = 1e-5
# print(bdust(100.,zavg))
# print(f_nu(100.))
# pl = io.Plotter(xlabel='',ylabel='')
# fnu = f_nu(freqs)
# bnu = bdust(freqs,zavg)
# pl.add(freqs,Y*fnu)
# pl.add(freqs,D*bnu)
# pl.hline()
# pl.done(io.dout_dir+"scaling.png")
