from useful import *
from plotter import plotClumpsPhotometry
from gal_extinction import Gal_Extinction

from photutils import CircularAperture, CircularAnnulus, aperture_photometry

def setupCatalog(trimcat):

    dtype = trimcat.dtype.descr

    _dtype = [("NPIX_APER", float), ("EXTINCT_EBV", float),("EXTINCT_AV", float)]

    for filt in filterset:
        _dtype.extend([("FLUX_{:s}".format(filt), float),
                       ("FLUXERR_{:s}".format(filt), float),
                       ("MAG_{:s}".format(filt), float),
                       ("MAGERR_{:s}".format(filt), float),
                       ("ORGFLUX_{:s}".format(filt), float),
                       ("ORGFLUXERR_{:s}".format(filt), float),
                       ("ORGMAG_{:s}".format(filt), float),
                       ("ORGMAGERR_{:s}".format(filt), float),
                       ("UNDFLUX_{:s}".format(filt), float),
                       ("UNDFLUXERR_{:s}".format(filt), float),
                       ("UNDMAG_{:s}".format(filt), float),
                       ("UNDMAGERR_{:s}".format(filt), float),
                       ("DIFFLUX_{:s}".format(filt), float),
                       ("DIFFSTD_{:s}".format(filt), float),
                       ("IMGBKGD_{:s}".format(filt), float),
                       ("NEWFLUX_{:s}".format(filt), float),
                       ("NEWFLUXERR_{:s}".format(filt), float),
                       ("NEWMAG_{:s}".format(filt), float),
                       ("NEWMAGERR_{:s}".format(filt), float),
                       ("GALFLUXAVG_{:s}".format(filt), float),
                       ("GALFLUXAVGERR_{:s}".format(filt), float),
                       ("EXTINCT_{:s}".format(filt), float)])

    catalog = np.recarray(len(trimcat),dtype=dtype+_dtype)
    for x in  dtype: catalog[x[0]] = trimcat[x[0]]
    for x in _dtype: catalog[x[0]] = -99
    return catalog

def applyAperCorr(catalog,objname):

    apersize = getClumpApertureSize(objname=objname)
    aperCorr = getApertureCorr(psf=apersize["psf_avg"],aper=0.5*apersize["phot"])

    for filt in filterset:
        for x in ["ORGFLUX","UNDFLUX","FLUX","NEWFLUX"]:
            catalog["{:s}_{:s}".format(x,filt)] *= aperCorr
            catalog["{:s}ERR_{:s}".format(x,filt)] *= aperCorr
    return catalog

def removeGalacticExtinction(catalog):

    gal_ext = Gal_Extinction()
    catalog['EXTINCT_EBV'] = gal_ext.calc_EBV(catalog['GAL_RA'],catalog['GAL_DEC'])
    catalog['EXTINCT_AV'] = gal_ext.calc_Av(ebv=catalog['EXTINCT_EBV'])

    for filt in filterset:

        catalog["EXTINCT_{:s}".format(filt)] = gal_ext.calc_Alambda(filt=filt,Av=catalog["EXTINCT_AV"])[0]

        for x in ["ORGFLUX","UNDFLUX","FLUX","NEWFLUX"]:
            catalog["{:s}_{:s}".format(x,filt)]    = gal_ext.remove_gal_ext(flux=catalog["{:s}_{:s}".format(x,filt)],   filt=filt, Av=catalog["EXTINCT_AV"])
            catalog["{:s}ERR_{:s}".format(x,filt)] = gal_ext.remove_gal_ext(flux=catalog["{:s}ERR_{:s}".format(x,filt)],filt=filt, Av=catalog["EXTINCT_AV"])

        catalog["DIFFLUX_{:s}".format(filt)]    = gal_ext.remove_gal_ext(flux=catalog["DIFFLUX_{:s}".format(filt)],   filt=filt, Av=catalog["EXTINCT_AV"])
        catalog["DIFFSTD_{:s}".format(filt)]    = gal_ext.remove_gal_ext(flux=catalog["DIFFSTD_{:s}".format(filt)],   filt=filt, Av=catalog["EXTINCT_AV"])
        catalog["IMGBKGD_{:s}".format(filt)]    = gal_ext.remove_gal_ext(flux=catalog["IMGBKGD_{:s}".format(filt)],   filt=filt, Av=catalog["EXTINCT_AV"])
        catalog["GALFLUXAVG_{:s}".format(filt)] = gal_ext.remove_gal_ext(flux=catalog["GALFLUXAVG_{:s}".format(filt)],filt=filt, Av=catalog["EXTINCT_AV"])
        catalog["GALFLUXAVGERR_{:s}".format(filt)] = gal_ext.remove_gal_ext(flux=catalog["GALFLUXAVGERR_{:s}".format(filt)],filt=filt, Av=catalog["EXTINCT_AV"])

    return catalog

def calcMagnitudes(catalog):

    for filt in filterset:
        for x in ["ORGFLUX","UNDFLUX","FLUX","NEWFLUX"]:

            y = x.replace("FLUX","MAG")
            flux = catalog["{:s}_{:s}".format(x,filt)]
            fluxerr = catalog["{:s}ERR_{:s}".format(x,filt)]

            cond = (flux > 0)
            catalog["{:s}_{:s}".format(y,filt)][cond]    = (-2.5 * np.log10(flux[cond]) + zeropoint[filt])
            catalog["{:s}ERR_{:s}".format(y,filt)][cond] = ( 2.5 / np.log(10) * (fluxerr[cond] / flux[cond]))

    return catalog

def convertFluxTouJy(catalog):

    for filt in filterset:

        fluxScale = calcFluxScale(zp0=23.9,zp1=zeropoint[filt])

        for x in ["ORGFLUX","UNDFLUX","FLUX","NEWFLUX"]:

            catalog["{:s}_{:s}".format(x,filt)] /= fluxScale
            catalog["{:s}ERR_{:s}".format(x,filt)] /= fluxScale

        catalog["DIFFLUX_{:s}".format(filt)] /= fluxScale
        catalog["DIFFSTD_{:s}".format(filt)] /= fluxScale
        catalog["IMGBKGD_{:s}".format(filt)] /= fluxScale
        catalog["GALFLUXAVG_{:s}".format(filt)] /= fluxScale
        catalog["GALFLUXAVGERR_{:s}".format(filt)] /= fluxScale

    return catalog

def measureClumpPhotometry(objname, destdir):

    savename = "photom/{1:s}/{0:s}_phot.fits".format(objname, destdir)
    catname  = "photom/{1:s}/{0:s}_trim.fits".format(objname, destdir)
    ssegname = "photom/smooth/{0:s}_smooth_seg.fits".format(objname)

    try:
        catalog = fitsio.getdata(catname)
    except OSError:
        catalog = None

    if catalog is not None:

        catalog = setupCatalog(catalog)
        apersize = getClumpApertureSize(objname=objname)
        apertures = getClumpApertures(catalog,objname=objname)
        clump_mask = getClumpMask(catalog,imshape=fitsio.getdata(ssegname).shape,radius=0.5*apersize["mask"])
        clump_mask2 = getClumpMask(catalog,imshape=fitsio.getdata(ssegname).shape,radius=0.5*apersize["mask"])
        catalog["NPIX_APER"] = apertures["aper"].area

        for j,filt in enumerate(filterset):

            imgname = "fits/{0:s}_{1:s}_Stamp.fits".format(objname, filt)
            img, img_hdr = fitsio.getdata(imgname, header=True)

            errname = "fits/{0:s}_{1:s}_Stamp.err.fits".format(objname, filt)
            err = fitsio.getdata(errname)

            ### Fix for exptime
            img /= getExpTime(objname=objname,filt=filt)
            err /= getExpTime(objname=objname,filt=filt)

            ### Clump photometry
            photom = aperture_photometry(img, apertures["aper"], error=err, method="subpixel", subpixels=5)
            catalog["ORGFLUX_{:s}".format(filt)] = photom["aperture_sum"]
            catalog["ORGFLUXERR_{:s}".format(filt)] = photom["aperture_sum_err"]

            ### Diffuse galaxy light calc
            mask_img = img.copy()
            mask_img[clump_mask] = -99.0
            for i, mask in enumerate(apertures["annl"].to_mask(method="center")):
                annulus = mask.multiply(mask_img)
                annulus = annulus[(annulus != -99.0) & (mask.data != 0)]
                if len(annulus) < 10:
                    raise Warning("Diffuse galaxy light determined using less than 10px for {:d} - {:s} - clump {:d}.".format(objname, filt, i+1))
                catalog["DIFFLUX_{:s}".format(filt)][i] = np.median(annulus)
                catalog["DIFFSTD_{:s}".format(filt)][i] = np.std(annulus)

            ### Calc the underlying galaxy light within aperture
            catalog["UNDFLUX_{:s}".format(filt)]    = catalog["DIFFLUX_{:s}".format(filt)] * catalog["NPIX_APER"]
            catalog["UNDFLUXERR_{:s}".format(filt)] = catalog["DIFFSTD_{:s}".format(filt)] * np.sqrt(catalog["NPIX_APER"])

            ### Clip when the underlying galaxy light is zero/negative or brighter than the aperture flux itself
            cond1 = (catalog["UNDFLUX_{:s}".format(filt)] <= 0)
            catalog["UNDFLUX_{:s}".format(filt)][cond1] = 0
            cond2 = (catalog["UNDFLUX_{:s}".format(filt)] >= catalog["ORGFLUX_{:s}".format(filt)])
            catalog["UNDFLUX_{:s}".format(filt)][cond2] = catalog["ORGFLUX_{:s}".format(filt)][cond2]

            ### Calc the clump flux by subtracting the galaxy flux
            catalog["FLUX_{:s}".format(filt)] = catalog["ORGFLUX_{:s}".format(filt)] - catalog["UNDFLUX_{:s}".format(filt)]
            catalog["FLUXERR_{:s}".format(filt)] = catalog["ORGFLUXERR_{:s}".format(filt)]

            ### Save image background level
            catalog["IMGBKGD_{:s}".format(filt)] = calcImgBackground(img=img,sseg=fitsio.getdata(ssegname))

            ### Gal sub attempt
            galseg = fitsio.getdata("photom/galaxy/{:s}_seg.fits".format(objname))
            galidx = galseg[int(galseg.shape[0]/2),int(galseg.shape[1]/2)]
            mask_img = img.copy()
            mask_img[clump_mask2 | (galseg!=galidx)] = -99.0
            galflux = mask_img[mask_img!=-99]

            catalog["GALFLUXAVG_{:s}".format(filt)], catalog["GALFLUXAVGERR_{:s}".format(filt)] = np.median(galflux), np.std(galflux)
            print(catalog["GALFLUXAVG_{:s}".format(filt)], catalog["GALFLUXAVGERR_{:s}".format(filt)], catalog["GALFLUXAVG_{:s}".format(filt)]/ catalog["GALFLUXAVGERR_{:s}".format(filt)])
            catalog["NEWFLUX_{:s}".format(filt)] = catalog["ORGFLUX_{:s}".format(filt)] - catalog["GALFLUXAVG_{:s}".format(filt)] * catalog["NPIX_APER"]
            catalog["NEWFLUXERR_{:s}".format(filt)] = catalog["ORGFLUXERR_{:s}".format(filt)]

        # catalog = applyAperCorr(catalog,objname=objname)
        catalog = removeGalacticExtinction(catalog)
        catalog = calcMagnitudes(catalog)
        catalog = convertFluxTouJy(catalog)

        fitsio.writeto(savename, catalog,overwrite=True)

    else:

        print("No clumps found for {:s}".format(objname))
        if os.path.isfile(savename): os.remove(savename)
        os.system("touch {:s}".format(savename))
        catalog = None

def main(sample):

    for j,objname in enumerate(sample):

        print("\rProcessing {:s} - [{:d}/{:d}] ... ".format(objname,j+1,len(sample)),end="",flush="")
        measureClumpPhotometry(objname=objname,destdir="boxcar8")
        plotClumpsPhotometry(objname=objname,destdir="boxcar8",savefig=True)

    print("done.")

def plot(sample):

    filelist = " ".join(["plots/photom/{0:s}_photom.png".format(i) for i in sample])
    savename="plots/pdf/clump_photometry.pdf"
    os.system("convert {0:s} {1:s}".format(filelist, savename))

if __name__ == "__main__":

    main(sample=sample)
    plot(sample=sample)

    plt.show()
