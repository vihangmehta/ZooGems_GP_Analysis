import os
import sys
import time
import subprocess
import numpy as np
import astropy.io.fits as fitsio
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Ellipse, Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

import scipy.optimize
import scipy.integrate
import scipy.interpolate
from photutils import CircularAperture, CircularAnnulus, aperture_photometry

cwd = "/data/highzgal/mehta/Misc/GP_Zoo_HST/"

sample = ["J0353-0010","J1004+2017","J1015+2227",
          "J1020+2937","J1055+0841","J1214+4520",
          "J1336+6255","J1504+5954","J1633+3753"]

sexaRaDecDict = {'J1336+6255': '13h36m07.9138s +62d55m30.77s',
                 'J1633+3753': '16h33m37.9414s +37d53m14.3s',
                 'J1055+0841': '10h55m30.4166s +08d41m32.9s',
                 'J1504+5954': '15h04m57.9874s +59d54m07.27s',
                 'J1214+4520': '12h14m23.1802s +45d20m40.91s',
                 'J1004+2017': '10h04m00.6406s +20d17m19.25s',
                 'J1015+2227': '10h15m41.1521s +22d27m27.52s',
                 'J0353-0010': '03h53m32.4636s -00d10m28.88s',
                 'J1020+2937': '10h20m57.4622s +29d37m26.47s'}

light = 3e18
FITS_pixscale = 0.05
filterset = ["F555W","F850LP"]
psf = {"F555W":0.1,"F850LP":0.1}
psf_avg = np.mean([psf[_] for _ in filterset])
zeropoint = {"F555W":25.711,"F850LP":24.856}
pivot = {"f555w":5360.95,"f850lp":9033.22}

def getRADec(objname):

    coords = SkyCoord(sexaRaDecDict[objname],frame="icrs")
    return coords.ra.deg, coords.dec.deg

def getZeroPoint(header):

    return -2.5*np.log10(a[1].header["PHOTFLAM"]) - 5*np.log10(a[1].header["PHOTPLAM"]) - 2.408

def getExpTime(objname,filt):

    return fitsio.getheader("fits/orig/{:s}_{:s}_Stamp.fits".format(objname,filt),0)["EXPTIME"]

def calcFluxScale(zp0,zp1):
    """
    -2.5*log(f1) + zp1 = -2.5*log(f0) + zp0
    f1/f0 = 10**((zp1 - zp0) / 2.5)
    """
    fscale = 10**((zp1 - zp0) / 2.5)
    return fscale

def calc_filter_flux(wave,spec,filt):

    filt_wave,filt_sens = np.genfromtxt("filters/acs_%s.pb"%(filt),unpack=True)
    filt_interp = scipy.interpolate.interp1d(filt_wave, filt_sens, bounds_error=False, fill_value=0, kind='linear')

    filt_sens = filt_interp(wave)
    if np.all(filt_sens==0): return 0
    flux = scipy.integrate.simps(spec*filt_sens*wave, wave) / scipy.integrate.simps(filt_sens*wave, wave)
    flux = (pivot[filt]**2/light) * flux
    return flux

def getClumpApertureSize(objname,units="pixel"):

    apersize = {"psf_avg": psf_avg,
                   "phot": psf_avg * 3.0,
                   "ann0": psf_avg * 4.0,
                   "ann1": psf_avg * 5.0,
                   "mask": psf_avg * 4.0,
                   "star": psf_avg * 6.0}

    if units=="arcsec":
        return apersize
    elif units=="pixel":
        for x in apersize: apersize[x] /= FITS_pixscale
        return apersize
    else:
        raise Exception("Invalid units for getClumpApertureSize -- choose between 'pixel' and 'arcsec'")

def getClumpPositions(catalog):

    return np.vstack([catalog["X"]-1, catalog["Y"]-1]).T

def getClumpApertures(catalog, objname):

    apersize = getClumpApertureSize(objname=objname)

    pos  = getClumpPositions(catalog)
    aper = CircularAperture(pos,  r=0.5*apersize["phot"])
    annl = CircularAnnulus(pos,r_in=0.5*apersize["ann0"],
                              r_out=0.5*apersize["ann1"])

    return {"aper":aper,"annl":annl}

def getClumpMask(catalog,imshape,radius):

    pos  = getClumpPositions(catalog)
    mask = np.zeros(imshape, dtype=bool)

    for (xc, yc) in pos:
        c, r = np.indices(mask.shape)
        cond = ((r - xc)**2 + (c - yc)**2 <= radius**2)
        mask[cond] = 1
    return mask

def gauss2d(x,y,sig,x0=0,y0=0):

    return 0.5/sig**2/np.pi * np.exp(-0.5*((x-x0)**2+(y-y0)**2)/sig**2)

def fwhmToSigma(fwhm):

    return fwhm / (2 * np.sqrt(2*np.log(2.)))

def getApertureCorr(psf,aper):

    dx = dy = 0.01
    x = np.arange(-10,10,dx)
    y = np.arange(-10,10,dy)

    yy,xx = np.meshgrid(y,x,sparse=True)
    g_see = gauss2d(xx,yy,sig=fwhmToSigma(psf))

    cond_aper = np.sqrt(xx**2 + yy**2) <= aper
    f_see = np.sum(g_see[cond_aper]) * dx * dy

    aper_adjust = 1/f_see
    return aper_adjust

def calcImgBackground(img,sseg):

    idx  = [int(np.floor(img.shape[1]*1/3)), int(np.ceil(img.shape[1]*2/3))]
    img  = img[ idx[0]:idx[1],idx[0]:idx[1]]
    sseg = sseg[idx[0]:idx[1],idx[0]:idx[1]]
    data = img[sseg==0]

    med, std = np.median(data),np.std(data)
    cond = (med-3*std<data) & (data<med+3*std)
    bckgd = np.std(data[cond])
    return bckgd

def calcCoM(img):
    """
    Calculate the center of mass for a given image
    """
    y, x = np.indices(img.shape) + 1
    xc = np.ma.sum(img * x) / np.ma.sum(img)
    yc = np.ma.sum(img * y) / np.ma.sum(img)
    return xc, yc

def calcReff(xc,yc,img,clumpMask,debug=False):

    minr, maxr = 0.5 * FITS_pixscale, np.sqrt(2) * max(img.shape) * FITS_pixscale
    radii = np.logspace(np.log10(minr), np.log10(maxr), int(np.ceil(maxr)))
    flux = np.zeros(len(radii))

    for i, (ri, ro) in enumerate(zip(radii[:-1], radii[1:])):
        aper = CircularAnnulus((xc - 1, yc - 1), r_in=ri, r_out=ro)
        mask = aper.to_mask(method="subpixel", subpixels=5)
        aimg = mask.multiply(img)
        aimg = aimg[(aimg >= 0) & (mask.data != 0)]
        if len(aimg) > 1:
            ### Only fill in the clumps, leave the seg edges alone
            new_area = len(aimg) + np.sum(mask.multiply(clumpMask))
            flux[i + 1] = flux[i] + np.mean(aimg) * new_area
        else:
            flux[i + 1] = flux[i]

    rcen = 0.5 * (radii[1:] + radii[:-1])
    flux = flux[1:] / np.max(flux)

    fint = scipy.interpolate.interp1d(rcen, flux, kind="cubic")
    reff = scipy.optimize.brentq(lambda x: fint(x) - 0.5, min(rcen), max(rcen) - 0.5)

    if debug:

        img = np.ma.masked_array(img, mask=img == -99)
        apers = [CircularAperture((xc - 1, yc - 1), r=r) for r in radii]
        _flux = aperture_photometry(img.filled(0), apers, method="subpixel", subpixels=5)
        _flux = [_flux["aperture_sum_{:d}".format(i)] for i in range(len(radii))]
        _flux = _flux / np.max(_flux)

        print("Reff:{:.2f}".format(reff))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=75, tight_layout=True)
        ax1.imshow(img, origin="lower")
        ax1.add_patch(Ellipse(xy=(xc-1,yc-1), width=2*reff, height=2*reff, angle=0, edgecolor='c', facecolor='none', lw=1))
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.plot(radii,_flux,c="k",marker="o",markersize=3,mew=0,label="w/o clump fill")
        ax2.plot(rcen,flux,c="tab:red",marker="o",markersize=3,mew=0,label="w/ clump fill")
        ax2.axhline(0.5, c="k", ls="--")
        ax2.axvline(reff, c="k", ls="--")
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_xlabel("Radius", fontsize=18)
        ax2.set_ylabel("Flux (<r) [norm]", fontsize=18)
        ax2.legend(fontsize=18)
        [_.set_fontsize(16) for _ in ax2.get_xticklabels() + ax2.get_yticklabels()]
        plt.show(block=True)

    return reff

def calcEllipseParameters(xc,yc,img,debug=False):

    y,x = np.indices(img.shape) + 1

    ### Second Moments
    x2 = np.ma.sum(img * x * x) / np.ma.sum(img) - xc**2
    y2 = np.ma.sum(img * y * y) / np.ma.sum(img) - yc**2
    xy = np.ma.sum(img * x * y) / np.ma.sum(img) - xc*yc

    sma = np.sqrt(0.5*(x2 + y2) + np.sqrt((0.5*(x2 - y2))**2 + xy**2))
    smb = np.sqrt(0.5*(x2 + y2) - np.sqrt((0.5*(x2 - y2))**2 + xy**2))
    theta = np.degrees(np.arctan(2*xy/(x2-y2)) / 2)

    if debug:

        print("SMA:{:.2f}, SMB:{:.2f}, THETA:{:.2f}".format(sma,smb,theta))
        fig,ax = plt.subplots(1,1,figsize=(6,6),dpi=75,tight_layout=True)
        ax.imshow(img,origin="lower")
        ax.add_patch(Ellipse(xy=(xc-1,yc-1), width=2*sma, height=2*smb, angle=theta, edgecolor='r', facecolor='none', lw=1))
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show(block=True)

    return sma,smb,theta

def getTotalNClumps(sample,destdir="boxcar8"):

    N = 0
    for objname in sample:
        try:
            N+=len(fitsio.getdata(os.path.join(cwd,"photom/{1:s}/{0:s}_trim.fits".format(objname, destdir))))
        except OSError:
            pass
    return N

def runBashCommand(call, cwd, verbose=True):
    """
    Generic function to execute a bash command
    """
    start = time.time()
    if isinstance(verbose, str):
        f = open(verbose, "w")
        p = subprocess.Popen(call, stdout=f, stderr=f, cwd=cwd, shell=True)
    elif verbose == True:
        print("Running command:<{:s}> in directory:<{:s}> ... ".format(call, cwd))
        p = subprocess.Popen(
            call, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd, shell=True
        )
        for line in iter(p.stdout.readline, b""):
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()
    else:
        devnull = open(os.devnull, "w")
        p = subprocess.Popen(call, stdout=devnull, stderr=devnull, cwd=cwd, shell=True)
    p.communicate()
    p.wait()
    return time.time() - start

