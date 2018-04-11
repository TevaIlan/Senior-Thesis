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
import healpy as hp

map90location='/Users/Teva/maps and catalog data/f090_night_all2_map_mono.fits'
map150location='/Users/Teva/maps and catalog data/f150_night_all2_map_mono.fits'
# map217location='/Users/Teva/maps and catalog data/HFI_SkyMap_217_2048_R2.02_full.fits'
# map353location='/Users/Teva/maps and catalog data/HFI_SkyMap_353_2048_R2.02_full.fits'
# map545location='/Users/Teva/maps and catalog data/HFI_SkyMap_545_2048_R2.02_full.fits'
# map857location='/Users/Teva/maps and catalog data/HFI_SkyMap_857_2048_R2.02_full.fits'


lmap90=enmap.read_map(map90location)
lmap150=enmap.read_map(map150location)
# lmap217=hp.read_map(map217location)
# lmap353=hp.read_map(map353location)
# lmap545=hp.read_map(map545location)
# lmap857=hp.read_map(map857location)

tmap90=lmap90[0]
tmap150=lmap150[0]
# tmap217=lmap217[0]
# tmap353=lmap353[0]
# tmap545=lmap545[0]
# tmap857=lmap857[0]
Ny,Nx = tmap90.shape #they have the same shape so it doesn't matter if we use the 90 or 150 map

#cat_location='../maps and catalog data/act_confirmed_clusters.fits'
#cat_location='../maps and catalog data/act_candidate_clusters.fits'
cat_location='../maps and catalog data/redmapper_dr8_public_v6.3_catalog.fits'
out_dir = "./"

hdulist=fits.open(cat_location)
catalog= hdulist[1].data

widthStampArcminute=60.
pixScale = 0.5
Np = np.int(widthStampArcminute/pixScale+0.5)
pad = np.int(Np/2+0.5)
N=0

#CalculatedY=[]
MeasuredY=[]
S90s=[]
S150s=[]
S217s=[]
S353s=[]
S545s=[]
S857s=[]
zvalues=[]
stack90=0
stack150=0
stack217=0
stack353=0

x90=((6.62607004*10**(-34)*90*10**9)/(1.38064852*10**(-23)*2.7255))
a90=10**-9*(x90*(np.cosh(x90/2.)/np.sinh(x90/2.))-4)*2.7255
#print(a90)
x150=((6.62607004*10**(-34)*150*10**9)/(1.38064852*10**(-23)*2.7255))
a150=10**-9*(x150*(np.cosh(x150/2.)/np.sinh(x150/2.))-4)*2.7255
#print(a150)
x217=((6.62607004*10**(-34)*217*10**9)/(1.38064852*10**(-23)*2.7255))
a217=10**-9*(x217*(np.cosh(x217/2.)/np.sinh(x217/2.))-4)*2.7255
#print(a217)
x353=((6.62607004*10**(-34)*353*10**9)/(1.38064852*10**(-23)*2.7255))
a353=10**-9*(x353*(np.cosh(x353/2.)/np.sinh(x353/2.))-4)*2.7255
#print(a353)

for i in range(0,len(catalog)):
	ra=catalog[i][2]*np.pi/180 #1 for ACT, 2 for redmapper
	dec=catalog[i][3]*np.pi/180 #2 for ACT, 3 for redmapper
	iy,ix = tmap90.sky2pix(coords=(dec,ra)) #doesn't matter if this uses 90 or 150
	if ix>=pad and ix<Nx-pad and iy>=pad and iy<Ny-pad:
		cutout90=maps.cutout(tmap90,arcmin_width=widthStampArcminute,ra=ra,dec=dec)
		cutout150=maps.cutout(tmap150,arcmin_width=widthStampArcminute,ra=ra,dec=dec)
		# cutout217=maps.cutout(tmap217,arcmin_width=widthStampArcminute,ra=ra,dec=dec)
		# cutout353=maps.cutout(tmap353,arcmin_width=widthStampArcminute,ra=ra,dec=dec)
		# cutout545=maps.cutout(tmap545,arcmin_width=widthStampArcminute,ra=ra,dec=dec)
		# cutout857=maps.cutout(tmap857,arcmin_width=widthStampArcminute,ra=ra,dec=dec)
		# S90=maps.aperture_photometry(instamp=cutout90,aperture_radius=6*np.pi/10800,annulus_width=(150/90)*1.4*np.pi/10800)
		# S90s.append(S90)
		# S150=maps.aperture_photometry(instamp=cutout150,aperture_radius=6*np.pi/10800,annulus_width=1.4*np.pi/10800)
		# S150s.append(S150)
		# S217=maps.aperture_photometry(instamp=cutout217,aperture_radius=17*np.pi/10800,annulus_width=5*np.pi/10800)
		# S217s.append(S217)
		# S353=maps.aperture_photometry(instamp=cutout353,aperture_radius=23*np.pi/10800,annulus_width=5*np.pi/10800)
		# S353s.append(S353)
		# S545=maps.aperture_photometry(instamp=cutout545,aperture_radius=23*np.pi/10800,annulus_width=5*np.pi/10800)
		# S545s.append(S545)
		# S857=maps.aperture_photometry(instamp=cutout857,aperture_radius=23*np.pi/10800,annulus_width=5*np.pi/10800)
		# S857s.append(S857)
		# # Ycalculated=(S90-S150)/(a90-a150)
		# # CalculatedY.append(Ycalculated)
		# #YMeasured=catalog[i][11]
		# #MeasuredY.append(YMeasured)
		# stack90=stack90+cutout90
		# stack150=stack150+cutout150
		# stack217=stack217+cutout217
		# stack353=stack353+cutout353
		z=catalog[i][4]
		zvalues.append(z)
		N=N+1
print(N)
print(np.mean(zvalues))
#print(np.max(MeasuredY), np.min(MeasuredY))
# frequencies=(90,150,217,353)
# Fluxes=np.mean(S90s),np.mean(S150s),np.mean(S217s),np.mean(S353s)
# z=np.mean(zvalues)
# def Sfunc(v,Y,D):
# 	h=6.62607004*10**(-34)
# 	k=1.38064852*10**(-23)
# 	Tcmb=2.7255
# 	c=299792458
# 	a=10**-9*(((h*v*10**9)/(k*Tcmb))*(np.cosh(((h*v*10**9)/(k*Tcmb))/2.)/np.sinh(((h*v*10**9)/(k*Tcmb))/2.))-4)*Tcmb
# 	b=(((v*10**9*(1+z))/353)**1.78*((2*h*(v*10**9*(1+z))**3)/(c**2)*(1/(np.exp((h*(v*10**9*(1+z)))/(k*20))-1))*(2*h**2*(v*10**9)**4*np.exp((h*(v*10**9*(1+z)))/(k*20)))/(c**2*k*20**2*(np.exp((h*(v*10**9*(1+z)))/(k*20))-1)**2))
# 	return a*Y+D*b
# popt, pcov=scipy.optimize.curve_fit(Sfunc,frequencies,Fluxes)
# print(popt)
# print(pcov)


# avalues=(a90,a150,a217,a353)
# Fluxes=np.mean(S90s),np.mean(S150s),np.mean(S217s),np.mean(S353s)
# yvalues=(np.mean(S90s),np.mean(S150s),np.mean(S217s),np.mean(S353s))
# error=(np.std(S90s)/np.sqrt(N),np.std(S150s)/np.sqrt(N),np.std(S217s)/np.sqrt(N),np.std(S353s)/np.sqrt(N))
# plt.errorbar(frequencies,yvalues,yerr=error,fmt='o')
# plt.errorbar(frequencies,avalues)
# plt.savefig('a values vs frequency')
# stack90=stack90/N
# stack150=stack150/N
# stack217=stack217/N
# stack353=stack353/N
# io.plot_img(stack90,"redmapper_stack90")
# io.plot_img(stack150,"redmapper_stack150")
# io.plot_img(stack217,"redmapper_stack217")
# io.plot_img(stack353,"redmapper_stack353")

# # print(S217s)
# # print(S353s)

# print('Flux 90:')
# print(np.mean(S90s)) 
# print('Error 90:') 
# print(np.std(S90s)/np.sqrt(N))

# print('Flux 150:')
# print(np.mean(S150s)) 
# print('Error 150:') 
# print(np.std(S150s)/np.sqrt(N))

# print('Flux 217:')
# print(np.mean(S217s)) 
# print('Error 217:') 
# print(np.std(S217s)/np.sqrt(N))

# print('Flux 353:')
# print(np.mean(S353s)) 
# print('Error 353:') 
# print(np.std(S353s)/np.sqrt(N))

frequencies=(90,150,217,353,545,857)
means=(np.mean(S90s),np.mean(S150s),np.mean(S217s),np.mean(S353s),np.mean(S545s),np.mean(S857s))
error=(np.std(S90s)/np.sqrt(N),np.std(S150s)/np.sqrt(N),np.std(S217s)/np.sqrt(N),np.std(S353s)/np.sqrt(N),np.std(S545s)/np.sqrt(N),np.std(S857s)/np.sqrt(N))
plt.errorbar(x=frequencies,y=means, yerr=error,fmt='o')
#plt.yscale('log')
plt.ylabel('Flux')
plt.xlabel('Frequencies')
plt.savefig('redmapper frequencies,fluxes, and errors')
plt.close()

# plt.scatter(MeasuredY,S90s)
# ymax=max(S90s)
# ymin=min(S90s)
# plt.ylim(ymin,ymax)
# plt.xlabel('Measured Y')
# plt.ylabel('Flux')
# plt.savefig("confirmed Flux vs Measured Y 90")
# plt.close()

# plt.scatter(MeasuredY,S150s)
# ymax=max(S150s)
# ymin=min(S150s)
# plt.ylim(ymin,ymax)
# plt.xlabel('Measured Y')
# plt.ylabel('Flux')
# plt.savefig("confirmed Flux vs Measured Y 150")
# plt.close()

# plt.scatter(MeasuredY,S217s)
# ymax=max(S217s)
# ymin=min(S217s)
# plt.ylim(ymin,ymax)
# plt.xlabel('Measured Y')
# plt.ylabel('Flux')
# plt.savefig("confirmed Flux vs Measured Y 217")
# plt.close()

# plt.scatter(MeasuredY,S353s)
# ymax=max(S353s)
# ymin=min(S353s)
# plt.ylim(ymin,ymax)
# plt.xlabel('Measured Y')
# plt.ylabel('Flux')
# plt.savefig("confirmed Flux vs Measured Y 353")
# plt.close()

# plt.hist(S90s)
# plt.savefig("confirmed Flux 90 Hist")
# plt.close()

# plt.hist(S150s)
# plt.savefig("confirmed Flux 150 Hist")
# plt.close()

# plt.hist(S217s)
# plt.savefig("confirmed Flux 217 Hist")
# plt.close()

# plt.hist(S353s)
# plt.savefig("confirmed Flux 353 Hist")
# plt.close()

# plt.scatter(MeasuredY,CalculatedY)
# ymax=max(CalculatedY)
# ymin=min(CalculatedY)
# plt.ylim(ymin,ymax)
# plt.xlabel('Measured Compton Y')
# plt.ylabel('Calculated Compton Y')
# plt.savefig("act_candidate RedMapper Calculated vs Measured Y")
# print(np.mean(CalculatedY), np.std(CalculatedY)/np.sqrt(N)) #prints (3.613133052108954e-11, 4.750134609152583e-12)
