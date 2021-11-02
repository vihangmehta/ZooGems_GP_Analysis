from useful import *
from plotter import plotClumpsCuration

from astropy.convolution import Box2DKernel,convolve_fft

def setupCatalog(sexcat,objname):

    dtype = [("ID", int),("X", float),("Y", float),("RA", float),("DEC", float),
             ("GAL_XC", float),("GAL_YC", float),("GAL_RA", float),("GAL_DEC", float),
             ("GAL_REFF", float),("GAL_SMA", float),("GAL_SMB", float),("GAL_THETA", float),
             ("GAL_REFF_XY", float),("GAL_SMA_XY", float),("GAL_SMB_XY", float),
             ("DISTANCE_XY", float),("DISTANCE", float),("DISTNORM", float),("DIST_SMA", float),
             ("PSF_WIDTH_AVG", float)]

    catalog = np.recarray(len(sexcat),dtype=dtype)
    for x in catalog.dtype.names: catalog[x] = -99

    catalog["ID"]  = sexcat["NUMBER"]
    catalog["X"]   = sexcat["X_IMAGE"]
    catalog["Y"]   = sexcat["Y_IMAGE"]
    # catalog["RA"]  = sexcat["X_WORLD"]
    # catalog["DEC"] = sexcat["Y_WORLD"]
    catalog["PSF_WIDTH_AVG"] = psf_avg

    catalog["GAL_RA"], catalog["GAL_DEC"] = getRADec(objname)

    return catalog

def mkSmoothSegMap(objname, filt="F850LP", smth=5, verbose=False):

    img = "fits/{0:s}_{1:s}_Stamp.fits".format(objname,filt)
    smoothed = "photom/smooth/{0:s}_{1:s}_smooth.fits".format(objname,filt)
    img, hdr = fitsio.getdata(img, header=True)

    ######################################################
    size = img.shape[0]
    __img = img[int(np.floor(size * 1.0 / 4.0)) : int(np.ceil(size * 3.0 / 4.0)),
                int(np.floor(size * 1.0 / 4.0)) : int(np.ceil(size * 3.0 / 4.0))]

    med, sig = np.median(__img), np.std(__img)
    img = np.clip(img, med - 50 * sig, med + 50 * sig)

    # med, sig = np.median(img), np.std(img)
    # _img = img[(med - 3 * sig < img) & (img < med + 3 * sig)]
    ######################################################

    kernel = Box2DKernel(smth)
    simg = convolve_fft(img, kernel, fill_value=np.NaN)
    fitsio.writeto(smoothed, data=simg, header=hdr, overwrite=True)

    args = {"det_img": "photom/smooth/{0:s}_{1:s}_smooth.fits".format(objname,filt),
            "catname": "photom/smooth/{0:s}_smooth_cat.fits".format(objname),
            "seg_img": "photom/smooth/{0:s}_smooth_seg.fits".format(objname),
            "detectThresh": 8.00,
            "analysisThresh": 8.01}

    call = "sex {det_img:s} " \
           "-c config/config_smooth.sex " \
           "-PARAMETERS_NAME config/param_smooth.sex " \
           "-CATALOG_NAME {catname:s} -CATALOG_TYPE FITS_1.0 " \
           "-WEIGHT_TYPE NONE " \
           "-DETECT_THRESH {detectThresh:.2f} -ANALYSIS_THRESH {analysisThresh:.2f} " \
           "-CHECKIMAGE_TYPE SEGMENTATION -CHECKIMAGE_NAME {seg_img:s} ".format(**args)

    runBashCommand(call, cwd=cwd, verbose=verbose)

def removeOutsideClumps(catalog, objname):
    """
    Remove clumps that are outside the parent galaxies' segmap
    """
    segimg = fitsio.getdata("photom/smooth/{0:s}_smooth_seg.fits".format(objname))
    segidx = segimg[int(np.round(segimg.shape[1] / 2)), int(np.round(segimg.shape[0] / 2))]

    if segidx != 0:
        cond = (segimg[np.round(catalog["Y"]-1).astype(int),
                       np.round(catalog["X"]-1).astype(int)]==segidx)
        catalog = catalog[cond]
    else:
        print("Segmap is zero at the center for ID#{0:s}".format(objname))
        catalog = catalog[np.zeros(len(catalog), dtype=bool)]

    return catalog

def getCoM(catalog, objname):
    """
    Calculate the CoM of the image after masking the detected clumps
    Also removes the CoM from the clump list, if it is marked as one
    """
    ### Get smoothed image, segmap and custom clump mask
    smth_img, smth_hdr = fitsio.getdata("photom/smooth/{0:s}_F850LP_smooth.fits".format(objname),header=True)
    smth_cat = fitsio.getdata("photom/smooth/{0:s}_smooth_cat.fits".format(objname))
    smth_seg = fitsio.getdata("photom/smooth/{0:s}_smooth_seg.fits".format(objname))
    segidx = smth_seg[int(np.round(smth_seg.shape[1] / 2)), int(np.round(smth_seg.shape[0] / 2))]

    apersize = getClumpApertureSize(objname)

    ### Get the mask for all detected clumps
    mask = getClumpMask(catalog,imshape=smth_img.shape,radius=0.5*apersize["mask"])

    ### Mask the clumps as well as everything that's not the galaxy in the smoothed image
    smth_img_masked = np.ma.masked_array(smth_img, mask=(smth_seg != segidx) | mask)

    ### Calculate the CoM
    catalog["GAL_XC"], catalog["GAL_YC"] = calcCoM(img=smth_img_masked)

    ### Convert the center (x,y) to (ra,dec)
    # wcs = WCS(smth_hdr, fix=False)
    # catalog["GAL_RA"], catalog["GAL_DEC"] = wcs.all_pix2world(catalog["GAL_XC"], catalog["GAL_YC"], 1)

    ### Calculate the distance from center
    catalog["DISTANCE_XY"] = np.sqrt((catalog["X"]-catalog["GAL_XC"])**2 +
                                     (catalog["Y"]-catalog["GAL_YC"])**2)
    catalog["DISTANCE"] = catalog["DISTANCE_XY"] * FITS_pixscale

    ### Don't count the CoM as a clump (presumably bulge)
    # catalog = catalog[catalog["DISTANCE"] > catalog["PSF_WIDTH_AVG"]]

    return catalog

def getMorphologyParameters(catalog,objname,filt="F850LP"):

    ### Get smoothed image, segmap and custom clump mask
    img = fitsio.getdata("fits/{0:s}_{1:s}_Stamp.fits".format(objname,filt))
    segimg = fitsio.getdata("photom/smooth/{0:s}_smooth_seg.fits".format(objname))
    segidx = segimg[int(np.round(segimg.shape[1] / 2)), int(np.round(segimg.shape[0] / 2))]

    ### Get the mask for all detected clumps
    apersize = getClumpApertureSize(objname)
    mask = getClumpMask(catalog,imshape=img.shape,radius=0.5*apersize["mask"])

    ### Mask the clumps as well as everything that's not the galaxy in the smoothed image
    img = np.ma.masked_array(img, mask=(segimg != segidx) | mask)

    catalog["GAL_REFF_XY"] = calcReff(xc=catalog["GAL_XC"][0],yc=catalog["GAL_YC"][0],img=img.filled(-99),clumpMask=mask)
    catalog["GAL_SMA_XY"], \
    catalog["GAL_SMB_XY"], \
    catalog["GAL_THETA"] = calcEllipseParameters(xc=catalog["GAL_XC"][0],yc=catalog["GAL_YC"][0],img=img)

    catalog["GAL_REFF"] = catalog["GAL_REFF_XY"] * FITS_pixscale
    catalog["GAL_SMA"]  = catalog["GAL_SMA_XY"]  * FITS_pixscale
    catalog["GAL_SMB"]  = catalog["GAL_SMB_XY"]  * FITS_pixscale
    catalog["DISTNORM"] = catalog["DISTANCE"] / catalog["GAL_REFF"]
    catalog["DIST_SMA"] = catalog["DISTANCE"] / catalog["GAL_SMA"]

    return catalog

def curateClumps(objname,destdir):

    catname  = "photom/{1:s}/{0:s}_cat.fits".format(objname, destdir)
    savename = "photom/{1:s}/{0:s}_trim.fits".format(objname, destdir)

    catalog = fitsio.getdata(catname)
    catalog = setupCatalog(catalog,objname=objname)
    catalog = removeOutsideClumps(catalog,objname=objname)

    if len(catalog)>0:
        catalog = getCoM(catalog,objname=objname)

    if len(catalog)>0:
        catalog = getMorphologyParameters(catalog,objname=objname)

    if len(catalog)>0:
        fitsio.writeto(savename, catalog, overwrite=True)
    else:
        print("No clumps for {0:s}".format(objname))
        if os.path.isfile(savename): os.remove(savename)
        os.system("touch {:s}".format(savename))

def main(sample):

    for j,objname in enumerate(sample):

        print("\rProcessing {:s} - [{:d}/{:d}] ... ".format(objname,j+1,len(sample)),end="",flush="")
        mkSmoothSegMap(objname,filt="F850LP",smth=10)
        curateClumps(objname=objname,destdir="boxcar8")
        plotClumpsCuration(objname=objname,destdir="boxcar8",savefig=True)

    print("done.")

def plot(sample):

    filelist = " ".join(["plots/curate/{0:s}_curate.png".format(i) for i in sample])
    savename="plots/pdf/clump_curation.pdf"
    os.system("convert {0:s} {1:s}".format(filelist, savename))

if __name__ == '__main__':

    main(sample=sample)
    plot(sample=sample)

    plt.show()
