from __future__ import print_function
from orphics import maps,io,cosmology,catalogs,stats,mpi
from enlib import enmap
import numpy as np
import os,sys
from szar import counts
import healpy as hp

def deltaTOverTcmbToJyPerSr(freqGHz,T0 = 2.7255):
    """
    @brief the function name is self-eplanatory
    @return the converstion factor
    """
    kB = 1.380658e-16
    h = 6.6260755e-27
    c = 29979245800.
    nu = freqGHz*1.e9
    x = h*nu/(kB*T0)
    cNu = 2*(kB*T0)**3/(h**2*c**2)*x**4/(4*(np.sinh(x/2.))**2)
    cNu *= 1e23
    return cNu


import argparse
parser = argparse.ArgumentParser(description='Stack and get AP on clusters.')
parser.add_argument("freq", type=str,help='Frequency tag. Supports 90/150/217/353.')
parser.add_argument("cat", type=str,help='Catalog tag. Supports redmapper/act_confirmed/act_candidate.')
parser.add_argument("--zmin",     type=float,  default=None,help="Minimum z. If specified, won't save AP.")
parser.add_argument("--zmax",     type=float,  default=None,help="Maximum z. If specified, won't save AP.")
parser.add_argument("--smin",     type=float,  default=None,help="Minimum S/N.")
parser.add_argument("--smax",     type=float,  default=None,help="Maximum S/N.")
parser.add_argument("--adaptive", action='store_true',help='AP based on 5*theta200.')
args = parser.parse_args()

zmin = args.zmin
zmax = args.zmax
smin = args.smin
smax = args.smax

freq = args.freq
io.dout_dir += "f"+freq+"_"+args.cat+"_"


if freq=="150":
    imap = enmap.read_fits('/Users/Teva/maps and catalog data/f150_night_all2_map_mono.fits')
    dmap = enmap.read_fits('/Users/Teva/maps and catalog data/f150_night_all2_div_mono.fits')
    apin = 6.
    apout = 1.5
    pt_cut = 1000
    ntheta = 1.

elif freq=="90":
    imap = enmap.read_fits('/Users/Teva/maps and catalog data/f090_night_all2_map_mono.fits')
    dmap = enmap.read_fits('/Users/Teva/maps and catalog data/f090_night_all2_div_mono.fits')    
    apin = 6.
    apout = 1.5*150./90.
    pt_cut = 1500
    ntheta = 1.
    
elif freq=="217":
    imap = hp.read_map('/Users/Teva/maps and catalog data/HFI_SkyMap_217_2048_R2.02_full_cutout_h0.fits')*1e6
    apin = 17.
    apout = 6.0
    pt_cut = 2000
    ntheta = 5.
    
elif freq=="353":
    imap = hp.read_map('/Users/Teva/maps and catalog data/HFI_SkyMap_353_2048_R2.02_full_cutout_h0.fits')*1e6
    apin = 23.
    apout = 6.0
    pt_cut = 15000
    ntheta = 5.

elif freq=="545":
    imap = hp.read_map('/Users/Teva/maps and catalog data/HFI_SkyMap_545_2048_R2.02_full.fits')*1e6*1e6/deltaTOverTcmbToJyPerSr(545.)*2.7255 
    apin = 23.
    apout = 6.0
    pt_cut = np.inf
    ntheta = 5.
    
elif freq=="857":
    imap = hp.read_map('/Users/Teva/maps and catalog data/HFI_SkyMap_857_2048_R2.02_full.fits')*1e6*1e6/deltaTOverTcmbToJyPerSr(857.)*2.7255 
    apin = 23.
    apout = 6.0
    pt_cut = np.inf
    ntheta = 5.


    
if args.cat=="redmapper":
    rmapper_file = '../maps and catalog data/redmapper_dr8_public_v6.3_catalog.fits'
    cols = catalogs.load_fits(rmapper_file,['RA','DEC','Z_LAMBDA','LAMBDA'],hdu_num=1,Nmax=None)
    ras = cols['RA']
    decs = cols['DEC']
    zs = cols['Z_LAMBDA']
    lams = cols['LAMBDA']
    act = False
elif args.cat=="act_confirmed":
    cfile = '../maps and catalog data/act_confirmed_clusters.fits'
    cols = catalogs.load_fits(cfile,['RAdeg','DECdeg','redshift','SNR'],hdu_num=1,Nmax=None)
    ras = cols['RAdeg']
    decs = cols['DECdeg']
    zs = cols['redshift']
    sns = cols['SNR']
    act = True
elif args.cat=="act_candidate":
    cfile = '../maps and catalog data/act_candidate_clusters.fits'
    cols = catalogs.load_fits(cfile,['RAdeg','DECdeg','SNR'],hdu_num=1,Nmax=None)
    ras = cols['RAdeg']
    decs = cols['DECdeg']
    zs = [0]*len(ras)
    sns = cols['SNR']
    act = True
    

cc = counts.ClusterCosmology(skipCls=True)


arcmin_width = 60.
px = 0.5
npix = int(arcmin_width/px)
shape,wcs = maps.rect_geometry(width_arcmin=arcmin_width,px_res_arcmin=px)
thetas = []



comm = mpi.MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()
Njobs = len(ras)
num_each,each_tasks = mpi.mpi_distribute(Njobs,numcores)
if rank==0: print ("At most ", max(num_each) , " tasks...")
my_tasks = each_tasks[rank]

pt_rad = (apin+apout)*np.pi/180./60.

st = stats.Stats(comm)
j = 0
for i,task in enumerate(my_tasks):
    #for i,(ra,dec,lam,z) in enumerate(zip(ras,decs,lams,zs)):
    ra = ras[task]
    dec = decs[task]
    z = zs[task]
    if z<zmin and (zmin is not None): continue
    if z>zmax and (zmax is not None): continue
    if act:
        sn = sns[task]
        if sn<smin and (smin is not None): continue
        if sn>smax and (smax is not None): continue
        

    if not(act) and args.adaptive:
        lam = lams[task]
        ftheta200 = ntheta*cc.theta200_from_richness(lam,z)
        apin = ftheta200*180.*60./np.pi
        pt_rad = (apin+apout)*np.pi/180./60.
        if (apin+apout)>arcmin_width/2.: continue
        
        
    # st.add_to_stats("thetas",np.array((ftheta200*60.*180./np.pi,)))

    
    if freq=="150" or freq=="90":
        ccut = maps.cutout(imap[0],arcmin_width,ra=np.deg2rad(ra),dec=np.deg2rad(dec))
        dcut = maps.cutout(dmap,arcmin_width,ra=np.deg2rad(ra),dec=np.deg2rad(dec))
        if (ccut is None) or (dcut is None): continue
    else:
        ccut = maps.cutout_gnomonic(imap,rot=(ra,dec),coord=['G','C'],
             xsize=npix,ysize=npix,reso=px,
             nest=False,remove_dip=False,
             remove_mono=False,gal_cut=0,
             flip='astro')
        ccut = enmap.enmap(ccut,wcs)
        assert np.all(ccut.shape==shape)
        dcut = np.array([[1.]])


    j += 1
    if j==1:
        modrmap = ccut.modrmap()

    if np.any(np.abs(ccut[modrmap<pt_rad])>pt_cut):
        st.add_to_stats("ptignored",np.array((1.,)))
        continue
    st.add_to_stats("count",np.array((1.,)))
    wt = dcut.mean()
    accut = ccut - ccut.mean()
    st.add_to_stack("stack",accut)
    st.add_to_stack("wstack",wt*accut)
    st.add_to_stats("wts",np.array((wt,)))
        

    apflux = maps.aperture_photometry(ccut,aperture_radius=apin*np.pi/180./60.,annulus_width=apout*np.pi/180./60.,modrmap=modrmap)
    st.add_to_stats("apflux",np.array((apflux,)))
    
    
    if rank==0 and ((task+1)%100)==0: print ("Rank 0 done with task ", task+1, " / " , len(my_tasks))

st.get_stacks()
st.get_stats()

if rank==0:

    N = st.vectors['count'].sum()
    print("Tot: ",N)
    try:
        print("Ptignored: ",st.vectors['ptignored'].sum())
    except:
        pass
        
    stack = st.stacks['stack']
    wstack = st.stacks['wstack']*N/st.vectors['wts'].sum()
    print(wstack.shape)
    
    # io.hist(st.vectors['thetas'],save_file = io.dout_dir+"thetahist.png")

    io.plot_img(stack,io.dout_dir+"stack.png")
    io.plot_img(wstack,io.dout_dir+"wstack.png")

    if (zmin is None) and (zmax is None):
        #np.savetxt("f"+freq+"_"+args.cat+"_apflux.txt",st.vectors['apflux'])
        io.save_cols("f"+freq+"_"+args.cat+"_apflux.txt",(st.vectors['apflux'].T,st.vectors['wts'].T))
