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
map217location='/Users/Teva/maps and catalog data/HFI_SkyMap_217_2048_R2.02_full_cutout_h0.fits'
map353location='/Users/Teva/maps and catalog data/HFI_SkyMap_353_2048_R2.02_full_cutout_h0.fits'

lmap90=enmap.read_map(map90location)
lmap150=enmap.read_map(map150location)
lmap217=enmap.read_map(map217location)
lmap353=enmap.read_map(map353location)
tmap90=lmap90[0]
tmap150=lmap150[0]
tmap217=lmap217[0]
tmap353=lmap353[0]
Ny,Nx = tmap90.shape #they have the same shape so it doesn't matter if we use the 90 or 150 map

cat_location='../maps and catalog data/act_confirmed_clusters.fits'
#cat_location='../maps and catalog data/act_candidate_clusters.fits'
#cat_location='../maps and catalog data/redmapper_dr8_public_v6.3_catalog.fits'
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
S90s15=[]
S90s3=[]
S90s45=[]
S90s6=[]
S90s75=[]
S90s8=[]
S90s95=[]
S150s15=[]
S150s3=[]
S150s45=[]
S150s6=[]
S150s75=[]
S150s8=[]
S150s95=[]

# S217s3=[]
# S217s5=[]
# S217s7=[]
# S217s9=[]
# S217s11=[]
# S217s13=[]
# S217s15=[]
# S353s3=[]
# S353s5=[]
# S353s7=[]
# S353s9=[]
# S353s11=[]
# S353s13=[]
# S353s15=[]
stack90=0
stack150=0
stack217=0
stack353=0
for i in range(0,len(catalog)):
	ra=catalog[i][1]*np.pi/180 #1 for ACT, 2 for redmapper
	dec=catalog[i][2]*np.pi/180 #2 for ACT, 3 for redmapper
	iy,ix = tmap90.sky2pix(coords=(dec,ra)) #doesn't matter if this uses 90 or 150
	if ix>=pad and ix<Nx-pad and iy>=pad and iy<Ny-pad:
		cutout90=maps.cutout(tmap90,arcmin_width=widthStampArcminute,ra=ra,dec=dec)
		cutout150=maps.cutout(tmap150,arcmin_width=widthStampArcminute,ra=ra,dec=dec)
		cutout217=maps.cutout(tmap217,arcmin_width=widthStampArcminute,ra=ra,dec=dec)
		cutout353=maps.cutout(tmap353,arcmin_width=widthStampArcminute,ra=ra,dec=dec)
		S9015=maps.aperture_photometry(instamp=cutout90,aperture_radius=(150/90)*6*np.pi/10800,annulus_width=(150/90)*1.4*np.pi/10800)
		S90s15.append(S9015)
		S903=maps.aperture_photometry(instamp=cutout90,aperture_radius=(150/90)*6*np.pi/10800,annulus_width=(150/90)*3*np.pi/10800)
		S90s3.append(S903)
		S9045=maps.aperture_photometry(instamp=cutout90,aperture_radius=(150/90)*6*np.pi/10800,annulus_width=(150/90)*4.5*np.pi/10800)
		S90s45.append(S9045)
		S906=maps.aperture_photometry(instamp=cutout90,aperture_radius=(150/90)*6*np.pi/10800,annulus_width=(150/90)*6*np.pi/10800)
		S90s6.append(S906)
		S9075=maps.aperture_photometry(instamp=cutout90,aperture_radius=(150/90)*6*np.pi/10800,annulus_width=(150/90)*7.5*np.pi/10800)
		S90s75.append(S9075)
		S908=maps.aperture_photometry(instamp=cutout90,aperture_radius=(150/90)*6*np.pi/10800,annulus_width=(150/90)*8*np.pi/10800)
		S90s8.append(S908)
		S9095=maps.aperture_photometry(instamp=cutout90,aperture_radius=(150/90)*6*np.pi/10800,annulus_width=(150/90)*9.5*np.pi/10800)
		S90s95.append(S9095)
		S15015=maps.aperture_photometry(instamp=cutout150,aperture_radius=6*np.pi/10800,annulus_width=1.4*np.pi/10800)
		S150s15.append(S15015)
		S1503=maps.aperture_photometry(instamp=cutout150,aperture_radius=6*np.pi/10800,annulus_width=3*np.pi/10800)
		S150s3.append(S1503)
		S15045=maps.aperture_photometry(instamp=cutout150,aperture_radius=6*np.pi/10800,annulus_width=4.5*np.pi/10800)
		S150s45.append(S15045)
		S1506=maps.aperture_photometry(instamp=cutout150,aperture_radius=6*np.pi/10800,annulus_width=6*np.pi/10800)
		S150s6.append(S1506)
		S15075=maps.aperture_photometry(instamp=cutout150,aperture_radius=6*np.pi/10800,annulus_width=7.5*np.pi/10800)
		S150s75.append(S15075)
		S1508=maps.aperture_photometry(instamp=cutout150,aperture_radius=6*np.pi/10800,annulus_width=8*np.pi/10800)
		S150s8.append(S1508)
		S15095=maps.aperture_photometry(instamp=cutout150,aperture_radius=6*np.pi/10800,annulus_width=9.5*np.pi/10800)
		S150s95.append(S15095)
		# S2173=maps.aperture_photometry(instamp=cutout217,aperture_radius=17*np.pi/10800,annulus_width=3*np.pi/10800)
		# S217s3.append(S2173)
		# S2175=maps.aperture_photometry(instamp=cutout217,aperture_radius=17*np.pi/10800,annulus_width=5*np.pi/10800)
		# S217s5.append(S2175)
		# S2177=maps.aperture_photometry(instamp=cutout217,aperture_radius=17*np.pi/10800,annulus_width=7*np.pi/10800)
		# S217s7.append(S2177)
		# S2179=maps.aperture_photometry(instamp=cutout217,aperture_radius=17*np.pi/10800,annulus_width=9*np.pi/10800)
		# S217s9.append(S2179)
		# S21711=maps.aperture_photometry(instamp=cutout217,aperture_radius=17*np.pi/10800,annulus_width=11*np.pi/10800)
		# S217s11.append(S21711)
		# S21713=maps.aperture_photometry(instamp=cutout217,aperture_radius=17*np.pi/10800,annulus_width=13*np.pi/10800)
		# S217s13.append(S21713)
		# S21715=maps.aperture_photometry(instamp=cutout217,aperture_radius=17*np.pi/10800,annulus_width=15*np.pi/10800)
		# S217s15.append(S21715)
		# S3533=maps.aperture_photometry(instamp=cutout353,aperture_radius=23*np.pi/10800,annulus_width=3*np.pi/10800)
		# S353s3.append(S3533)
		# S3535=maps.aperture_photometry(instamp=cutout353,aperture_radius=23*np.pi/10800,annulus_width=5*np.pi/10800)
		# S353s5.append(S3535)
		# S3537=maps.aperture_photometry(instamp=cutout353,aperture_radius=23*np.pi/10800,annulus_width=7*np.pi/10800)
		# S353s7.append(S3537)
		# S3539=maps.aperture_photometry(instamp=cutout353,aperture_radius=23*np.pi/10800,annulus_width=9*np.pi/10800)
		# S353s9.append(S3539)
		# S35311=maps.aperture_photometry(instamp=cutout353,aperture_radius=23*np.pi/10800,annulus_width=11*np.pi/10800)
		# S353s11.append(S35311)
		# S35313=maps.aperture_photometry(instamp=cutout353,aperture_radius=23*np.pi/10800,annulus_width=13*np.pi/10800)
		# S353s13.append(S35313)
		# S35315=maps.aperture_photometry(instamp=cutout353,aperture_radius=23*np.pi/10800,annulus_width=15*np.pi/10800)
		# S353s15.append(S35315)
		
		# x90=((6.62607004*10**(-34)*90*10**9)/(1.38064852*10**(-23)*2.7255))
		# a90=2.7255*(x90*(np.cosh(x90/2.)/np.sinh(x90/2.))-4)
		# x150=((6.62607004*10**(-34)*150*10**9)/(1.38064852*10**(-23)*2.7255))
		# a150=2.7255*(x150*(np.cosh(x150/2.)/np.sinh(x150/2.))-4)
		# Ycalculated=(S90-S150)/(a90-a150)
		# CalculatedY.append(Ycalculated)
		YMeasured=catalog[i][11]
		MeasuredY.append(YMeasured)
		stack90=stack90+cutout90
		stack150=stack150+cutout150
		stack217=stack217+cutout217
		stack353=stack353+cutout353
		N=N+1
print(N)
# print(np.max(MeasuredY), np.min(MeasuredY))
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

frequencies=(90,150)
#error=(np.std(S90s)/np.sqrt(N),np.std(S150s)/np.sqrt(N),np.std(S217s)/np.sqrt(N),np.std(S353s)/np.sqrt(N))
plt.errorbar(x=90-4,y=np.mean(S90s15), yerr=np.std(S90s15)/np.sqrt(N),fmt='o')
plt.errorbar(x=90-2,y=np.mean(S90s3), yerr=np.std(S90s3)/np.sqrt(N),fmt='o')
plt.errorbar(x=90+2,y=np.mean(S90s45), yerr=np.std(S90s45)/np.sqrt(N),fmt='o')
plt.errorbar(x=90+4,y=np.mean(S90s6), yerr=np.std(S90s6)/np.sqrt(N),fmt='o')
plt.errorbar(x=90+6,y=np.mean(S90s75), yerr=np.std(S90s75)/np.sqrt(N),fmt='o')
plt.errorbar(x=90+8,y=np.mean(S90s8), yerr=np.std(S90s8)/np.sqrt(N),fmt='o')
plt.errorbar(x=90+10,y=np.mean(S90s95), yerr=np.std(S90s95)/np.sqrt(N),fmt='o')
plt.errorbar(x=150-4,y=np.mean(S150s15), yerr=np.std(S150s15)/np.sqrt(N),fmt='o')
plt.errorbar(x=150-2,y=np.mean(S150s3), yerr=np.std(S150s3)/np.sqrt(N),fmt='o')
plt.errorbar(x=150+2,y=np.mean(S150s45), yerr=np.std(S150s45)/np.sqrt(N),fmt='o')
plt.errorbar(x=150+4,y=np.mean(S150s6), yerr=np.std(S150s6)/np.sqrt(N),fmt='o')
plt.errorbar(x=150+6,y=np.mean(S150s75), yerr=np.std(S150s75)/np.sqrt(N),fmt='o')
plt.errorbar(x=150+8,y=np.mean(S150s8), yerr=np.std(S150s8)/np.sqrt(N),fmt='o')
plt.errorbar(x=150+10,y=np.mean(S150s95), yerr=np.std(S150s95)/np.sqrt(N),fmt='o')

# plt.errorbar(x=217-6,y=np.mean(S217s3), yerr=np.std(S217s3)/np.sqrt(N),fmt='o')
# plt.errorbar(x=217-4,y=np.mean(S217s5), yerr=np.std(S217s5)/np.sqrt(N),fmt='o')
# plt.errorbar(x=217-2,y=np.mean(S217s7), yerr=np.std(S217s7)/np.sqrt(N),fmt='o')
# plt.errorbar(x=217,y=np.mean(S217s9), yerr=np.std(S217s9)/np.sqrt(N),fmt='o')
# plt.errorbar(x=217+2,y=np.mean(S217s11), yerr=np.std(S217s11)/np.sqrt(N),fmt='o')
# plt.errorbar(x=217+4,y=np.mean(S217s13), yerr=np.std(S217s13)/np.sqrt(N),fmt='o')
# plt.errorbar(x=217+6,y=np.mean(S217s15), yerr=np.std(S217s15)/np.sqrt(N),fmt='o')
# plt.errorbar(x=353-6,y=np.mean(S353s3), yerr=np.std(S353s3)/np.sqrt(N),fmt='o')
# plt.errorbar(x=353-4,y=np.mean(S353s5), yerr=np.std(S353s5)/np.sqrt(N),fmt='o')
# plt.errorbar(x=353-2,y=np.mean(S353s7), yerr=np.std(S353s7)/np.sqrt(N),fmt='o')
# plt.errorbar(x=353,y=np.mean(S353s9), yerr=np.std(S353s9)/np.sqrt(N),fmt='o')
# plt.errorbar(x=353+2,y=np.mean(S353s11), yerr=np.std(S353s11)/np.sqrt(N),fmt='o')
# plt.errorbar(x=353+4,y=np.mean(S353s13), yerr=np.std(S353s13)/np.sqrt(N),fmt='o')
# plt.errorbar(x=353+6,y=np.mean(S353s15), yerr=np.std(S353s15)/np.sqrt(N),fmt='o')



#plt.yscale('log')
plt.ylabel('Flux')
plt.xlabel('Frequencies')
plt.savefig('varying annulus 90 and 150')
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
