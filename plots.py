import enlib
import orphics
from enlib import enmap
from astropy.io import fits
import numpy as np
from orphics import maps
import orphics.io as io
from orphics.io import Plotter as pl
import matplotlib
import matplotlib.pyplot as plt
import szar
from szar import counts
import scipy
from orphics import maps,io,cosmology,catalogs,stats,mpi
ccon = cosmology.defaultConstants

map90location='/Users/Teva/maps and catalog data/f090_night_all2_map_mono.fits'
map150location='/Users/Teva/maps and catalog data/f150_night_all2_map_mono.fits'
map217location='/Users/Teva/maps and catalog data/HFI_SkyMap_217_2048_R2.02_full_cutout_h0.fits'
map353location='/Users/Teva/maps and catalog data/HFI_SkyMap_353_2048_R2.02_full_cutout_h0.fits'

#cat_location='../maps and catalog data/act_confirmed_clusters.fits'
#cat_location='../maps and catalog data/act_candidate_clusters.fits'
cat_location='../maps and catalog data/redmapper_dr8_public_v6.3_catalog.fits'
out_dir = "./"

hdulist=fits.open(cat_location)
catalog= hdulist[1].data
lmap90=enmap.read_map(map90location)
tmap90=lmap90[0]
Ny,Nx = tmap90.shape
widthStampArcminute=60.
pixScale = 0.5
Np = np.int(widthStampArcminute/pixScale+0.5)
pad = np.int(Np/2+0.5)
z_values=[]
richnesses=[]
N=0

for i in range(0,len(catalog)):
	ra=catalog[i][2]*np.pi/180 #1 for ACT, 2 for redmapper
	dec=catalog[i][3]*np.pi/180 #2 for ACT, 3 for redmapper
	iy,ix = tmap90.sky2pix(coords=(dec,ra)) #doesn't matter if this uses 90 or 150
	if ix>=pad and ix<Nx-pad and iy>=pad and iy<Ny-pad:
		z=catalog[i][4]
		richness=catalog[i][6]
		z_values.append(z)
		richnesses.append(richness)
		N=N+1

print(N)
# plt.hist(z_values)
# plt.xlabel('Z')
# plt.ylabel("Number of Clusters")
# plt.savefig('redmapper_z_hist.pdf')

plt.hist(richnesses, bins=20)
plt.xlabel('Richness')
plt.xlim(xmax=150)
plt.ylabel("Number of Clusters")
plt.savefig('redmapper_richness_hist.pdf')

freqs = np.arange(30,900,1.)
def f_nu(nu):
    nu = np.asarray(nu)
    mu = ccon['H_CGS']*(1e9*nu)/(ccon['K_CGS']*ccon['TCMB'])
    ans = mu/np.tanh(mu/2.0) - 4.0
    return ans
# print(freqs)
# for i in freqs:
# 	print(i)
# 	print(gfunc(i)) 
plt.plot(freqs,f_nu(freqs))
plt.ylabel(r'$g(\nu)$')
plt.xlabel(r'$\nu$')
plt.savefig('gnu.png')
plt.close()

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
    print('Hi')
    print(nu_ghz)
    print(np.asarray(nu_ghz))
    print(z)
    pref = (nu_ghz*(1+z)/nu0)**(beta)
    return pref * planck(nu_ghz*(1+z),Tdust) / dplanckT(nu_ghz)

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

plt.plot(freqs,gfuncrel(freqs*1e9,Te=0,),label='Teff=0')
plt.plot(freqs,gfuncrel(freqs*1e8,Te=1e8,),label='Teff=1e8')
plt.plot(freqs,gfuncrel(freqs*1e8,Te=1e10,),label='Teff=1e10')
#plt.plot(freqs,gfuncrel(freqs*1e9,Te=1e12,),label='Teff=1e12')
plt.ylabel(r'$g(\nu)$')
plt.xlabel(r'$\nu$')
plt.legend()
plt.savefig('gnu.png')

  
