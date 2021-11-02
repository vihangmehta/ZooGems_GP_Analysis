from useful import *
from gal_extinction import Gal_Extinction

def runSextractor(objname,verbose=False):

    imgs = dict([(filt,"fits/{0:s}_{1:s}_Stamp.fits".format(objname,filt)) for filt in filterset])
    errs = dict([(filt,"fits/{0:s}_{1:s}_Stamp.err.fits".format(objname,filt)) for filt in filterset])

    for filt in filterset:

        data,hdr = fitsio.getdata(imgs[filt],header=True)
        imgs[filt] = "photom/galaxy/{0:s}_{1:s}.fits".format(objname,filt)
        fitsio.writeto(imgs[filt],data,header=hdr,overwrite=True)

        data,hdr = fitsio.getdata(errs[filt],header=True)
        errs[filt] = "photom/galaxy/{0:s}_{1:s}.err.fits".format(objname,filt)
        fitsio.writeto(errs[filt],data,header=hdr,overwrite=True)

    detimg = np.average([fitsio.getdata(imgs[_]) for _ in filterset],axis=0)
    fitsio.writeto("photom/galaxy/{0:s}_det.fits".format(objname),detimg,overwrite=True)

    for filt in filterset:

        args = {"sci_img": imgs[filt],
                "sci_wht": errs[filt],
                "det_img": "photom/galaxy/{0:s}_det.fits".format(objname),
                "det_wht": "NONE",
                "catname": "photom/galaxy/{0:s}_{1:s}_cat.fits".format(objname,filt),
                "seg_img": "photom/galaxy/{0:s}_seg.fits".format(objname),
                "zp": zeropoint[filt]} #getZeroPoint(fitsio.getheader(imgs[filt]))}

        call = "sex {det_img:s},{sci_img:s} " \
               "-c config/config_galaxy.sex " \
               "-PARAMETERS_NAME config/param_galaxy.sex " \
               "-CATALOG_NAME {catname:s} -CATALOG_TYPE FITS_1.0 -MAG_ZEROPOINT {zp:.4f} " \
               "-WEIGHT_TYPE NONE,MAP_RMS -WEIGHT_IMAGE {det_wht:s},{sci_wht:s} " \
               "-CHECKIMAGE_TYPE SEGMENTATION -CHECKIMAGE_NAME {seg_img:s}".format(**args)

        runBashCommand(call, cwd=cwd, verbose=verbose)

def removeGalacticExtinction(catalog):

    gal_ext = Gal_Extinction()
    catalog['EXTINCT_EBV'] = gal_ext.calc_EBV(catalog['RA'],catalog['DEC'])
    catalog['EXTINCT_AV'] = gal_ext.calc_Av(ebv=catalog['EXTINCT_EBV'])

    for filt in filterset:
        catalog["EXTINCT_{:s}".format(filt)] = gal_ext.calc_Alambda(filt=filt,Av=catalog["EXTINCT_AV"])[0]
        for x in ["AUTO","ISO"]:
            catalog[   "FLUX_{:s}_{:s}".format(x,filt)] = gal_ext.remove_gal_ext(flux=catalog[   "FLUX_{:s}_{:s}".format(x,filt)],filt=filt,Av=catalog["EXTINCT_AV"])
            catalog["FLUXERR_{:s}_{:s}".format(x,filt)] = gal_ext.remove_gal_ext(flux=catalog["FLUXERR_{:s}_{:s}".format(x,filt)],filt=filt,Av=catalog["EXTINCT_AV"])

    return catalog

def calcMagnitudes(catalog):

    for filt in filterset:
        for x in ["AUTO","ISO"]:
            flux    = catalog[   "FLUX_{:s}_{:s}".format(x,filt)]
            fluxerr = catalog["FLUXERR_{:s}_{:s}".format(x,filt)]
            cond = (flux > 0)
            catalog[   "MAG_{:s}_{:s}".format(x,filt)][cond] = (-2.5 * np.log10(flux[cond]) + zeropoint[filt])
            catalog["MAGERR_{:s}_{:s}".format(x,filt)][cond] = ( 2.5 / np.log(10) * (fluxerr[cond] / flux[cond]))

    return catalog

def convertFluxTouJy(catalog):

    for filt in filterset:
        fluxScale = calcFluxScale(zp0=23.9,zp1=zeropoint[filt])
        for x in ["AUTO","ISO"]:
            catalog[   "FLUX_{:s}_{:s}".format(x,filt)] /= fluxScale
            catalog["FLUXERR_{:s}_{:s}".format(x,filt)] /= fluxScale
    return catalog

def mkGalaxyPhotom():

    dtype = [("ID","U10"),("RA",float),("DEC",float),("EXTINCT_EBV",float),("EXTINCT_AV",float)]
    for filt in filterset:
        for aper in ["AUTO","ISO"]:
            dtype += [(   "FLUX_%s_%s"%(aper,filt),float),
                      ("FLUXERR_%s_%s"%(aper,filt),float),
                      (    "MAG_%s_%s"%(aper,filt),float),
                      ( "MAGERR_%s_%s"%(aper,filt),float)]
        dtype += [("EXTINCT_%s"%filt,float)]
    gal_cat = np.recarray(len(sample),dtype=dtype)

    for i,objname in enumerate(sample):

        print("\rProcessing {:d}/{:d} ... ".format(i+1,len(sample)),end="",flush=True)

        clump_cat = fitsio.getdata("photom/boxcar8/{0:s}_phot.fits".format(objname))
        COMx,COMy = clump_cat["GAL_XC"][0],clump_cat["GAL_YC"][0]

        segimg = fitsio.getdata("photom/galaxy/{0:s}_seg.fits".format(objname))
        segidx = segimg[int(np.round(COMy-1)), int(np.round(COMx-1))]

        if segidx==0:
            print(objname,"SEGMAP is 0 at stamp center!")
            plt.imshow(segimg,origin="lower")
            plt.scatter(COMx-1,COMy-1,color="w",lw=2,marker="x")
            plt.show()

        gal_cat["ID"][i] = objname
        gal_cat["RA"][i], gal_cat["DEC"][i] = getRADec(objname)

        for filt in filterset:

            cat = fitsio.getdata("photom/galaxy/{0:s}_{1:s}_cat.fits".format(objname,filt))
            cat_objname = cat[segidx-1]

            for aper in ["AUTO","ISO"]:
                gal_cat[   "FLUX_%s_%s"%(aper,filt)][i] = cat_objname["flux_%s"%aper] / getExpTime(objname=objname,filt=filt)
                gal_cat["FLUXERR_%s_%s"%(aper,filt)][i] = cat_objname["fluxerr_%s"%aper] / getExpTime(objname=objname,filt=filt)

    gal_cat = removeGalacticExtinction(gal_cat)
    gal_cat = calcMagnitudes(gal_cat)
    gal_cat = convertFluxTouJy(gal_cat)

    print("done.")
    fitsio.writeto("photom/galaxy/galaxy_photom.fits",gal_cat,overwrite=True)

def chkSegmaps():

    from plotter import getVminVmax, reprocessSegMap

    for filt in list(filterset)+["seg","smthseg"]:

        print("\rPlotting {:s} ... ".format(filt),end="")
        fig,axes = plt.subplots(3,3,figsize=(20,20),dpi=75)
        fig.subplots_adjust(left=0,right=1,top=1,bottom=0,hspace=0,wspace=0)
        axes = axes.flatten()

        for i,(ax,objname) in enumerate(zip(axes,sample)):

            if "seg" not in filt:
                img = fitsio.getdata("photom/galaxy/{:s}_{:s}.fits".format(objname,filt))
                vmin,vmax = getVminVmax(img)
                ax.imshow(img,origin="lower",vmin=vmin,vmax=vmax,cmap=plt.cm.Greys)
            elif filt=="seg":
                img = fitsio.getdata("photom/galaxy/{:s}_seg.fits".format(objname))
                img = reprocessSegMap(img)
                ax.imshow(img,origin="lower",cmap=plt.cm.CMRmap,vmin=1,vmax=np.max(img))
            elif filt=="smthseg":
                img = fitsio.getdata("photom/smooth/{:s}_smooth_seg.fits".format(objname))
                img = reprocessSegMap(img)
                ax.imshow(img,origin="lower",cmap=plt.cm.CMRmap,vmin=1,vmax=np.max(img))

            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

        for ax in axes[i+1:]:
            ax.set_visible(False)

        fig.savefig("plots/galaxy/stamps_{:s}.png".format(filt))
        plt.close(fig)

    print("done.")

def main():

    for i,objname in enumerate(sample):
        print("\rProcessing {:d}/{:d} ... ".format(i+1,len(sample)),end="")
        runSextractor(objname=objname,verbose=False)
    print("done.")

if __name__ == '__main__':

    main()

    mkGalaxyPhotom()
    chkSegmaps()

    plt.show()
