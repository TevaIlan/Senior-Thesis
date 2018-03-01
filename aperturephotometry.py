import enlib
import orphics
from enlib import enmap
from astropy.io import fits
import numpy as np
from orphics import maps

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

for i in range(0,len(catalog)):
	ra=catalog[i][1]*np.pi/180
	dec=catalog[i][2]*np.pi/180
	iy,ix = tmap90.sky2pix(coords=(dec,ra)) #doesn't matter if this uses 90 or 150
	if ix>=pad and ix<Nx-pad and iy>=pad and iy<Ny-pad:
		cutout90=maps.cutout(tmap90,arcmin_width=widthStampArcminute,ra=ra,dec=dec)
		cutout150=maps.cutout(tmap150,arcmin_width=widthStampArcminute,ra=ra,dec=dec)
		S90=maps.aperture_photometry(instamp=cutout90,aperture_radius=1.5*np.pi/10800,annulus_width=(150/90)*1.4*np.pi/10800)
		S150=maps.aperture_photometry(instamp=cutout150,aperture_radius=1.5*np.pi/10800,annulus_width=1.4*np.pi/10800)
		N=N+1
print(N)