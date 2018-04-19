import numpy as np

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
