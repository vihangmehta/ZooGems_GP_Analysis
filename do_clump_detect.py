from useful import *

from astropy.convolution import Gaussian2DKernel,Box2DKernel,Tophat2DKernel,convolve_fft
from plotter import plotClumpsDetection

def detectClumps(objname,smth=10,kernel_type="boxcar",destdir="boxcar10",verbose=False):

    kernel = Box2DKernel(smth)

    imgs = {}
    for filt in filterset:

        img = "fits/orig/{0:s}_{1:s}_Stamp.fits".format(objname,filt)
        smoothed = "photom/{2:s}/{0:s}_{1:s}_smooth.fits".format(objname,filt,destdir)
        contrast = "photom/{2:s}/{0:s}_{1:s}_cntrst.fits".format(objname,filt,destdir)
        filtered = "photom/{2:s}/{0:s}_{1:s}_filter.fits".format(objname,filt,destdir)

        img, hdr = fitsio.getdata(img, header=True)

        ################################################################
        ##### FIX for crazy bright stars right next to main object #####
        ################################################################
        size = img.shape[0]
        __img = img[int(np.floor(size / 4.0)) : int(np.ceil(size * 3.0 / 4.0)),
                    int(np.floor(size / 4.0)) : int(np.ceil(size * 3.0 / 4.0))]

        ### Just clip the outliers
        img = np.clip(img,np.median(img) - 10 * np.std(__img),
                          np.median(img) + 10 * np.std(__img))
        ################################################################

        med, sig = np.median(img), np.std(img)
        _img = img[(med - 3 * sig < img) & (img < med + 3 * sig)]
        bckgrnd = np.median(_img)

        simg = convolve_fft(img, kernel, fill_value=bckgrnd)
        cimg = img - simg

        fimg = cimg.copy()
        med, std = np.median(fimg), np.std(fimg)
        _fimg = fimg[(med - 3 * sig < fimg) & (fimg < med + 3 * sig)]
        fimg[(fimg < (np.median(_fimg) + 2 * np.std(_fimg)))] = 0

        fitsio.writeto(smoothed, data=simg, header=hdr, overwrite=True)
        fitsio.writeto(contrast, data=cimg, header=hdr, overwrite=True)
        fitsio.writeto(filtered, data=fimg, header=hdr, overwrite=True)

        imgs[filt] = filtered

    detimg = np.sum([fitsio.getdata(imgs[_]) for _ in filterset],axis=0)
    fitsio.writeto("photom/{1:s}/{0:s}_det.fits".format(objname,destdir),detimg,overwrite=True)

    args = {"det_img": "photom/{1:s}/{0:s}_det.fits".format(objname,destdir),
            "catname": "photom/{1:s}/{0:s}_cat.fits".format(objname,destdir),
            "seg_img": "photom/{1:s}/{0:s}_seg.fits".format(objname,destdir)}

    call = "sex {det_img:s} " \
           "-c config/config_clump_detect.sex " \
           "-PARAMETERS_NAME config/param_clump_detect.sex " \
           "-CATALOG_NAME {catname:s} -CATALOG_TYPE FITS_1.0 " \
           "-WEIGHT_TYPE NONE " \
           "-CHECKIMAGE_TYPE SEGMENTATION -CHECKIMAGE_NAME {seg_img:s}".format(**args)

    runBashCommand(call, cwd=cwd, verbose=verbose)

def main(sample):

    for j,objname in enumerate(sample):
        print("\rProcessing {:s} - [{:d}/{:d}] ... ".format(objname,j+1,len(sample)),end="",flush="")
        detectClumps(objname,smth= 4,kernel_type="boxcar",destdir="boxcar4")
        detectClumps(objname,smth= 6,kernel_type="boxcar",destdir="boxcar6")
        detectClumps(objname,smth= 8,kernel_type="boxcar",destdir="boxcar8")
        detectClumps(objname,smth=10,kernel_type="boxcar",destdir="boxcar10")
        plotClumpsDetection(objname,savefig=True)
    print("done.")

def plot(sample):

    filelist = " ".join(["plots/detect/{0:s}_clumps.png".format(i) for i in sample])
    savename="plots/pdf/clump_detection.pdf"
    os.system("convert {0:s} {1:s}".format(filelist, savename))

def chkStats(sample):

    bins = 10 ** np.arange(-5, 5, 0.01)
    binc = 0.5 * (bins[1:] + bins[:-1])

    for destdir,c in zip(["boxcar4",  "boxcar6", "boxcar8", "boxcar10"],
                         ["tab:blue", "tab:red", "tab:green", "orange"]):

        counts_det = np.zeros(0)
        counts_filt = np.zeros(0)

        for i, objname in enumerate(sample):

            img = fitsio.getdata("photom/{1:s}/{0:s}_det.fits".format(objname,destdir))
            counts_det = np.append(counts_det, img[img != 0])

            for filt in filterset:
                img = fitsio.getdata("photom/{2:s}/{0:s}_{1:s}_filter.fits".format(objname,filt,destdir))
                counts_filt = np.append(counts_filt, img[img != 0])

        hist = np.histogram(counts_det, bins=bins)[0]
        plt.plot(binc, hist, color=c, lw=1.5, ls="-", alpha=0.6, label=destdir)

        hist = np.histogram(counts_filt, bins=bins)[0]
        plt.plot(binc, hist, color=c, lw=1.5, ls="--", alpha=0.6)

    plt.legend()
    plt.xscale("log")

if __name__ == '__main__':

    main(sample=sample)
    plot(sample=sample)

    chkStats(sample=sample)

    plt.show()
