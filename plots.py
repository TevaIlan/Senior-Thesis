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
plt.hist(z_values)
plt.xlabel('Z')
plt.ylabel("Number of Clusters")
plt.savefig('redmapper z hist.png')

plt.hist(richnesses)
plt.xlabel('Richness')
plt.ylabel("Number of Clusters")
plt.savefig('redmapper richness hist.png')