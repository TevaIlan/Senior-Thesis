from __future__ import print_function
from __future__ import print_function
from orphics import maps,io,cosmology,catalogs,stats,mpi
from enlib import enmap
import numpy as np
import os,sys
from szar import counts,foregrounds


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
def Sflux(fghz,Yin,Din):
    # if Yin<0: return np.nan
    # if Din<0: return np.nan
    return yflux(fghz,Yin)+dflux(fghz,Din)#+dT

from scipy.optimize import curve_fit

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
x = np.array([90,150,217,353,545,857])
aperrs=np.array(aperrs)
apmeans=np.array(apmeans)
bfit,bcov,chisquare,pte = stats.fit_linear_model(x,apmeans,ycov=np.diag(aperrs**2.), funcs=[lambda x: yflux(x,1.),lambda x: dflux(x,1.)])
Yfit,Dfit = bfit
eYfit,eDfit = np.sqrt(np.diagonal(bcov))


#print(dT,Y,D)
print(Yfit,Dfit)
print(Yfit/eYfit,Dfit/eDfit)
print(chisquare,pte)
print(bcov)
bcor=stats.cov2corr(bcov)
print(bcor)
# print(freqs)
# print(Yfit[0])
# print(Dfit[0])
# print(dTfit[0])
# print(Sflux(freqs,Yfit[0],Dfit[0],dTfit[0]))
# print(np.abs(Sflux(freqs,Yfit[0],Dfit[0],dTfit[0])))

pl = io.Plotter(xlabel='$\\nu$ (GHz)',ylabel='Flux ($K-\mathrm{arcmin}^2$)',yscale='log')
pl.add(freqs,np.abs(yflux(freqs,Yfit[0])),label='Y')
pl.add(freqs,dflux(freqs,Dfit[0]),label="D")
pl.add(freqs,np.abs(Sflux(freqs,Yfit[0],Dfit[0])),label='Fit')
pl.add_err(x,np.abs(apmeans),yerr=aperrs,marker="o",ls="none")
pl.legend()
pl.done(io.dout_dir+"apfluxes_fitlog.pdf")

print(apmeans)
print(aperrs)
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
