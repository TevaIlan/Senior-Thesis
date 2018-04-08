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

map90location='/Users/Teva/maps and catalog data/f090_daynight_all_map_mono_deep56.fits'
map150location='/Users/Teva/maps and catalog data/f150_daynight_all_map_mono_deep56.fits'
map217location='/Users/Teva/maps and catalog data/HFI_SkyMap_217_2048_R2.02_full_cutout_h0.fits'
map353location='/Users/Teva/maps and catalog data/HFI_SkyMap_353_2048_R2.02_full_cutout_h0.fits'