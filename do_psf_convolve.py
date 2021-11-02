from useful import *
from astropy.convolution import convolve

def mk_kernel_pypher(srcfilt,tarfilt,psfdir,pixscale,reg=1e-5):

    srcPSF = "psf{:s}Reduced.fits".format(srcfilt)
    tarPSF = "psf{:s}Reduced.fits".format(tarfilt)
    kernel = "ker_{:s}to{:s}.fits".format(srcfilt.upper(),tarfilt.upper())

    if os.path.isfile(os.path.join(psfdir,kernel)):
        os.remove(os.path.join(psfdir,kernel))

    runBashCommand(call="addpixscl {:s} {:f}".format(srcPSF,pixscale),cwd=psfdir)
    runBashCommand(call="addpixscl {:s} {:f}".format(tarPSF,pixscale),cwd=psfdir)
    runBashCommand(call="pypher {:s} {:s} {:s} -r {:.0e}".format(srcPSF,tarPSF,kernel,reg),cwd=psfdir)

    os.remove("{:s}/{:s}".format(psfdir,kernel.replace(".fits",".log")))

def mk_PSF_conv_plot(psfdir):

    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm,LogNorm
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(10,12),dpi=75,gridspec_kw={"height_ratios":[3,2]},tight_layout=True)
    vmin,vmax,dv = -3,3,0.25
    bins = np.arange(-50,50,dv)

    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    ax1.set_aspect(1.)

    tPSF = fitsio.getdata("{:s}/psff850lpReduced.fits".format(psfdir))
    oPSF = fitsio.getdata("{:s}/psff555wReduced.fits".format(psfdir))
    kern = fitsio.getdata("{:s}/ker_F555WtoF850LP.fits".format(psfdir))
    cPSF = convolve(oPSF,kern,boundary="fill",fill_value=0)

    sub = (cPSF - tPSF) / tPSF * 100
    im = ax1.imshow(sub,origin="lower",cmap=plt.cm.RdBu_r,vmin=vmin,vmax=vmax)

    ax_divider = make_axes_locatable(ax1)
    cbaxes = ax_divider.append_axes("right", size="4%", pad="1%")
    cbax = fig.colorbar(mappable=im, cax=cbaxes, orientation='vertical')

    y,x = np.indices(sub.shape)
    rad = np.sqrt((x-sub.shape[1]/2)**2 + (y-sub.shape[0]/2)**2)
    wht = 1 / (rad+1)
    cond = (rad<50)
    ax2.hist(sub[cond],bins=bins,weights=wht[cond],color='k',alpha=0.7)

    ax2.axvline(0,lw=0.5,ls='--',color='k')
    ax2.set_yscale("log")
    ax2.set_xlim(-10,10)
    ax2.set_title("(PSF$_{F555W,conv}$ - PSF$_{F850LP}$) / PSF$_{F850LP}$",fontsize=18)

    plt.show()

def convolve_stamps(sample):

    kern = fitsio.getdata("psfs/ker_F555WtoF850LP.fits")

    for objname in sample:

        img,hdr = fitsio.getdata("fits/orig/{:s}_F555W_Stamp.fits".format(objname),header=True)
        img = convolve(img,kern,boundary="fill",fill_value=0)
        fitsio.writeto("fits/{:s}_F555W_Stamp.fits".format(objname),img,header=hdr,overwrite=True)

        img,hdr = fitsio.getdata("fits/orig/{:s}_F850LP_Stamp.fits".format(objname),header=True)
        fitsio.writeto("fits/{:s}_F850LP_Stamp.fits".format(objname),img,header=hdr,overwrite=True)

        err,hdr = fitsio.getdata("fits/orig/{:s}_F555W_Stamp.fits".format(objname),ext=2,header=True)
        fitsio.writeto("fits/{:s}_F555W_Stamp.err.fits".format(objname),err,header=hdr,overwrite=True)
        err,hdr = fitsio.getdata("fits/orig/{:s}_F850LP_Stamp.fits".format(objname),ext=2,header=True)
        fitsio.writeto("fits/{:s}_F850LP_Stamp.err.fits".format(objname),err,header=hdr,overwrite=True)

if __name__ == '__main__':

    mk_kernel_pypher(srcfilt="f555w",tarfilt="f850lp",psfdir="psfs/",pixscale=0.05)
    mk_PSF_conv_plot(psfdir="psfs/")

    convolve_stamps(sample=sample)
