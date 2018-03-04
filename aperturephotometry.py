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

map90location='/Users/Teva/maps and catalog data/f090_daynight_all_map_mono_deep56.fits'
map150location='/Users/Teva/maps and catalog data/f150_daynight_all_map_mono_deep56.fits'

lmap90=enmap.read_map(map90location)
lmap150=enmap.read_map(map150location)
tmap90=lmap90[0]
tmap150=lmap150[0]
Ny,Nx = tmap90.shape #they have the same shape so it doesn't matter if we use the 90 or 150 map

cat_location='../maps and catalog data/act_confirmed_clusters.fits'
#cat_location='act_candidate_clusters.fits'
out_dir = "./"

hdulist=fits.open(cat_location)
catalog= hdulist[1].data

widthStampArcminute=60.
pixScale = 0.5
Np = np.int(widthStampArcminute/pixScale+0.5)
pad = np.int(Np/2+0.5)
N=0

CalculatedY=[]
MeasuredY=[]
S90s=[]
S150s=[]
stack90=0
stack150=0
for i in range(0,len(catalog)):
	ra=catalog[i][1]*np.pi/180
	dec=catalog[i][2]*np.pi/180
	iy,ix = tmap90.sky2pix(coords=(dec,ra)) #doesn't matter if this uses 90 or 150
	if ix>=pad and ix<Nx-pad and iy>=pad and iy<Ny-pad:
		cutout90=maps.cutout(tmap90,arcmin_width=widthStampArcminute,ra=ra,dec=dec)
		cutout150=maps.cutout(tmap150,arcmin_width=widthStampArcminute,ra=ra,dec=dec)
		S90=maps.aperture_photometry(instamp=cutout90,aperture_radius=1.5*np.pi/10800,annulus_width=(150/90)*1.4*np.pi/10800)
		S90s.append(S90)
		S150=maps.aperture_photometry(instamp=cutout150,aperture_radius=1.5*np.pi/10800,annulus_width=1.4*np.pi/10800)
		S150s.append(S150)
		x90=((6.62607004*10**-34*90*10**9)/(1.38064852*10**-23*2.7255))
		a90=2.7255*x90*(np.cosh(x90/2.)/np.sinh(x90/2.))
		x150=((6.62607004*10**-34*150*10**9)/(1.38064852*10**-23*2.7255))
		a150=2.7255*x150*(np.cosh(x150/2.)/np.sinh(x150/2.))
		Ycalculated=(S90-S150)/(a90-a150)
		CalculatedY.append(Ycalculated)
		YMeasured=catalog[i][11]
		MeasuredY.append(YMeasured)
		stack90=stack90+cutout90
		stack150=stack150+cutout150
		N=N+1
print(N)

stack90=stack90/N
stack150=stack150/N
io.plot_img(stack90,"stack90")
io.plot_img(stack150,"stack150")

plt.scatter(MeasuredY,S90s)
ymax=max(S90s)
ymin=min(S90s)
plt.ylim(ymin,ymax)
plt.xlabel('Measured Y')
plt.ylabel('Flux')
plt.savefig("Flux vs Measured Y 90")
plt.close()

plt.scatter(MeasuredY,S150s)
ymax=max(S150s)
ymin=min(S150s)
plt.ylim(ymin,ymax)
plt.xlabel('Measured Y')
plt.ylabel('Flux')
plt.savefig("Flux vs Measured Y 150")
plt.close()

plt.hist(S90s)
plt.savefig("Flux 90 Hist")
plt.close()

plt.hist(S150s)
plt.savefig("Flux 150 Hist")
plt.close()

plt.scatter(MeasuredY,CalculatedY)
ymax=max(CalculatedY)
ymin=min(CalculatedY)
plt.ylim(ymin,ymax)
plt.xlabel('Measured Compton Y')
plt.ylabel('Calculated Compton Y')
plt.savefig("Calculated vs Measured Y")
