from useful import *

def getVminVmax(img,sigclip=False):

    size = img.shape[0]
    _img = img[int(np.floor(size / 3.0)) : int(np.ceil(size * 2.0 / 3.0)),
               int(np.floor(size / 3.0)) : int(np.ceil(size * 2.0 / 3.0))]
    _img = np.ma.masked_array(_img, mask=~np.isfinite(_img))
    if sigclip:
        _img = _img[(_img<np.ma.median(_img)+3*np.ma.std(_img)) & \
                    (_img>np.ma.median(_img)-3*np.ma.std(_img))]
    vmin = np.ma.median(_img) - 1.0*np.ma.std(_img)
    vmax = np.ma.median(_img) + 2.0*np.ma.std(_img)
    return vmin, vmax

def reprocessSegMap(segm):
    """
    Reprocesses a segementation map for easier plotting
    """
    iobj = segm[int(segm.shape[0] / 2), int(segm.shape[1] / 2)]
    uniq = np.sort(np.unique(segm))
    if iobj == 0:
        iobj = np.max(uniq) + 1
    uniq = uniq[(uniq != iobj)]
    uniq = np.append(uniq, iobj)
    renumber_dict = dict(zip(uniq, np.arange(len(uniq)) + 1))
    _segm = segm.copy()
    for _i in uniq:
        _segm[segm == _i] = renumber_dict[_i]
    return _segm

def plotClumpsDetection(objname,savefig=True):

    figsize = np.array([13,4]) * 1.5
    fig = plt.figure(figsize=figsize, dpi=150)
    fig.subplots_adjust( left=  0.05/figsize[0]*figsize[1],
                        right=1-0.05/figsize[0]*figsize[1],
                        bottom=0.03,top=0.865)
    fig.suptitle("{:s}".format(objname),fontsize=22,fontweight=600)

    ogs = fig.add_gridspec(1,5,width_ratios=[1,3,3,3,3],wspace=0.025,hspace=0.1)
    igs0 = ogs[0].subgridspec(3,1,wspace=0,hspace=0,height_ratios=[2,2,3])
    igs1 = ogs[1].subgridspec(3,6,wspace=0,hspace=0,height_ratios=[2,2,3])
    igs2 = ogs[2].subgridspec(3,6,wspace=0,hspace=0,height_ratios=[2,2,3])
    igs3 = ogs[3].subgridspec(3,6,wspace=0,hspace=0,height_ratios=[2,2,3])
    igs4 = ogs[4].subgridspec(3,6,wspace=0,hspace=0,height_ratios=[2,2,3])

    for i, filt in enumerate(filterset):

        img = fitsio.getdata("fits/{0:s}_{1:s}_Stamp.fits".format(objname,filt))
        vmin, vmax = getVminVmax(img)

        ax = fig.add_subplot(igs0[i])
        ax.imshow(img,cmap=plt.cm.Greys,vmin=vmin,vmax=vmax,origin="lower",rasterized=True)
        ax.set_xlim(img.shape[0]*0.05,img.shape[0]*0.95)
        ax.set_ylim(img.shape[1]*0.05,img.shape[1]*0.95)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        if i==0: ax.set_title("SCI", fontsize=16, fontweight=600)

    for _igs,destdir in zip([igs1,igs2,igs3,igs4],["boxcar4","boxcar6","boxcar8","boxcar10"]):

        for i,filt in enumerate(filterset):

            for j,label in enumerate(["smooth","cntrst","filter"]):

                img = fitsio.getdata("photom/{3:s}/{0:s}_{1:s}_{2:s}.fits".format(objname,filt,label,destdir))
                vmin, vmax = getVminVmax(img)

                ax = fig.add_subplot(_igs[i,2*j:2*j+2])
                ax.imshow(img,cmap=plt.cm.Greys,vmin=vmin,vmax=vmax,origin="lower",rasterized=True)
                ax.set_xlim(img.shape[0]*0.05,img.shape[0]*0.95)
                ax.set_ylim(img.shape[1]*0.05,img.shape[1]*0.95)
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)

                if i==0 and j==1: ax.set_title(destdir, fontsize=14, fontweight=600)

        img = fitsio.getdata("photom/{1:s}/{0:s}_det.fits".format(objname,destdir))
        vmin, vmax = getVminVmax(img)
        ax = fig.add_subplot(_igs[i+1,:3])
        ax.imshow(img,cmap=plt.cm.Greys,vmin=vmin,vmax=vmax,origin="lower")
        ax.set_xlim(img.shape[0]*0.05,img.shape[0]*0.95)
        ax.set_ylim(img.shape[1]*0.05,img.shape[1]*0.95)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        segm = fitsio.getdata("photom/{1:s}/{0:s}_seg.fits".format(objname,destdir))
        segm = reprocessSegMap(segm)
        ax = fig.add_subplot(_igs[i+1,3:])
        ax.imshow(segm,cmap=plt.cm.CMRmap,vmin=1,vmax=np.max(segm),origin="lower")
        ax.set_xlim(img.shape[0]*0.05,img.shape[0]*0.95)
        ax.set_ylim(img.shape[1]*0.05,img.shape[1]*0.95)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    if savefig:
        fig.savefig("plots/detect/{0:s}_clumps.png".format(objname))
        plt.close(fig)
    else:
        return fig

def plotClumpsCuration(objname,destdir,savefig=True):

    fullcat = fitsio.getdata("photom/{1:s}/{0:s}_cat.fits".format(objname, destdir))
    try:
        trimcat = fitsio.getdata("photom/{1:s}/{0:s}_trim.fits".format(objname, destdir))
    except OSError:
        trimcat = None

    fig,[ax1,ax2,ax3,ax4] = plt.subplots(1,4,figsize=(15,4.2),dpi=120)
    fig.subplots_adjust(left=0,right=1,top=0.9,bottom=0,wspace=0,hspace=0)
    fig.suptitle("{:s} ({:s})".format(objname,destdir),fontsize=22,fontweight=600)

    segimg = fitsio.getdata("photom/{1:s}/{0:s}_seg.fits".format(objname,destdir))
    segimg = reprocessSegMap(segimg)
    ax3.imshow(segimg,cmap=plt.cm.CMRmap,vmin=1,vmax=np.max(segimg),origin="lower")
    ax3.text(0.02,0.98,"SEGM",color="lawngreen",fontsize=20,fontweight=600,va="top",ha="left",transform=ax3.transAxes)

    segimg = fitsio.getdata("photom/smooth/{0:s}_smooth_seg.fits".format(objname))
    segimg = reprocessSegMap(segimg)
    ax4.imshow(segimg,cmap=plt.cm.CMRmap,vmin=1,vmax=np.max(segimg),origin="lower")
    ax4.text(0.02,0.98,"SEGM (smooth)",color="lawngreen",fontsize=20,fontweight=600,va="top",ha="left",transform=ax4.transAxes)

    for ax,filt in zip([ax1,ax2],filterset):

        img,img_hdr = fitsio.getdata("fits/{0:s}_{1:s}_Stamp.fits".format(objname,filt),header=True)
        vmin,vmax = getVminVmax(img)
        ax.imshow(img,origin="lower",cmap=plt.cm.Greys,vmin=vmin,vmax=vmax)
        ax.text(0.02,0.98,filt,color="k",fontsize=20,fontweight=600,va="top",ha="left",transform=ax.transAxes)

    for ax in [ax1,ax2,ax3,ax4]:

        if trimcat is not None:
            rejcat = fullcat[~np.in1d(fullcat["NUMBER"],trimcat["ID"])]
        else:
            rejcat = fullcat
        ax.scatter(rejcat["X_IMAGE"]-1,rejcat["Y_IMAGE"]-1,s=30,color="r",marker="x",lw=1)
        if trimcat is not None:
            ax.scatter(trimcat["X"]-1,trimcat["Y"]-1,s=30,color="lawngreen",marker="x",lw=1)

        if trimcat is not None:
            # Mark the computed CoM for the diffuse light
            ax.scatter(trimcat["GAL_XC"][0]-1,trimcat["GAL_YC"][0]-1,s=250,color="lawngreen",marker="+",lw=3)
            # Add a circle showing the petro radius
            ax.add_patch(Circle(xy=(trimcat["GAL_XC"][0]-1, trimcat["GAL_YC"][0]-1),
                                radius=trimcat["GAL_REFF_XY"][0],
                                facecolor="none",edgecolor="lawngreen",lw=1,ls="--"))
            # Add a ellipse showing the morph fit
            # ax.add_patch(Ellipse(xy=(trimcat["GAL_XC"][0]-1, trimcat["GAL_YC"][0]-1),
            #                      width=2*trimcat["GAL_SMA_XY"][0], height=2*trimcat["GAL_SMB_XY"][0],
            #                      angle=trimcat["GAL_THETA"][0],
            #                      edgecolor='lawngreen',facecolor='none',lw=1,ls="--"))

    for ax in [ax1,ax2,ax3,ax4]:
        ax.set_xlim(segimg.shape[0]*0.05,segimg.shape[0]*0.95)
        ax.set_ylim(segimg.shape[1]*0.05,segimg.shape[1]*0.95)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    if savefig:
        fig.savefig("plots/curate/{0:s}_curate.png".format(objname))
        plt.close(fig)
    else:
        return fig

def plotClumpsPhotometry(objname,destdir,photcat=None,savefig=True):

    fullcat = fitsio.getdata("photom/{1:s}/{0:s}_cat.fits".format(objname, destdir))
    if photcat is None:
        try:
            photcat = fitsio.getdata("photom/{1:s}/{0:s}_phot.fits".format(objname, destdir))
        except OSError:
            photcat = None
    elif len(photcat)==0:
        photcat = None

    fig, axes = plt.subplots(2,3,figsize=(9,6.5),dpi=75)
    fig.subplots_adjust(left=0.02/3,right=1-0.02/3,top=0.92,bottom=0.02/2,wspace=0,hspace=0)
    fig.suptitle("ID {:s} ({:s})".format(objname,destdir),fontsize=22,fontweight=600)

    segimg = fitsio.getdata("photom/{1:s}/{0:s}_seg.fits".format(objname,destdir))
    segimg = reprocessSegMap(segimg)
    axes[0,0].imshow(segimg,cmap=plt.cm.CMRmap,vmin=1,vmax=np.max(segimg),origin="lower")
    axes[0,0].text(0.05,0.95,"SEGM",ha="left",va="top",color="lime",fontsize=20,fontweight=800,transform=axes[0,1].transAxes)

    segimg = fitsio.getdata("photom/smooth/{0:s}_smooth_seg.fits".format(objname))
    segimg = reprocessSegMap(segimg)
    axes[1,0].imshow(segimg,cmap=plt.cm.CMRmap,vmin=1,vmax=np.max(segimg),origin="lower")
    axes[1,0].text(0.05,0.95,"SEGM (smooth)",ha="left",va="top",color="lime",fontsize=20,fontweight=800,transform=axes[1,1].transAxes)

    if photcat is not None:
        apersize = getClumpApertureSize(objname=objname)
        apertures = getClumpApertures(photcat,objname=objname)
        clump_mask = getClumpMask(photcat,imshape=segimg.shape,radius=0.5*apersize["mask"])

    for j,filt in enumerate(filterset):

        imgname = "fits/{0:s}_{1:s}_Stamp.fits".format(objname,filt)
        img,img_hdr = fitsio.getdata(imgname,header=True)

        vmin,vmax = getVminVmax(img)
        axes[0,j+1].text(0.05,0.95,filt,ha="left",va="top",color="purple",fontsize=24,fontweight=800,transform=axes[0,j+1].transAxes)
        axes[0,j+1].imshow(img,origin="lower",cmap=plt.cm.Greys,vmin=vmin,vmax=vmax)

        if photcat is not None:
            mask_img = img.copy()
            mask_img[clump_mask] = -99.0
            axes[1,j+1].imshow(mask_img,origin="lower",cmap=plt.cm.Greys,vmin=vmin,vmax=vmax)
            apertures["annl"].plot(axes=axes[1,j+1],color="blue",lw=0.5)

    _axes = np.append(axes[0,1:],axes[1,1])
    if photcat is None:
        for ax in _axes:
            ax.scatter(fullcat["X_IMAGE"]-1,fullcat["Y_IMAGE"]-1,s=30,color="r",marker="x",lw=1)
    else:
        for ax in np.append(_axes,axes[1,2:]):
            # Mark all rejected "clumps"
            rejcat = fullcat[~np.in1d(fullcat["NUMBER"],photcat["ID"])]
            ax.scatter(rejcat["X_IMAGE"]-1,rejcat["Y_IMAGE"]-1,s=30,color="r",marker="x",lw=1)
            # Mark all clumps
            ax.scatter(photcat["X"]-1,photcat["Y"]-1,s=30,color="lawngreen",marker="x",lw=1)

            # Mark the computed CoM for the diffuse light
            ax.scatter(photcat["GAL_XC"][0]-1,photcat["GAL_YC"][0]-1,s=250,color="lawngreen",marker="+",lw=3)
            # Add a circle showing the petro radius
            ax.add_patch(Circle(xy=(photcat["GAL_XC"][0]-1, photcat["GAL_YC"][0]-1),
                                radius=photcat["GAL_REFF_XY"][0],
                                facecolor="none",edgecolor="lawngreen",lw=1,ls="--"))
            # Add a ellipse showing the morph fit
            # ax.add_patch(Ellipse(xy=(photcat["GAL_XC"][0]-1, photcat["GAL_YC"][0]-1),
            #                      width=2*photcat["GAL_SMA_XY"][0], height=2*photcat["GAL_SMB_XY"][0],
            #                      angle=photcat["GAL_THETA"][0],
            #                      edgecolor='lawngreen',facecolor='none',lw=1,ls="--"))


    for ax in axes.flatten():
        ax.set_xlim(segimg.shape[0]*0.05,segimg.shape[0]*0.95)
        ax.set_ylim(segimg.shape[1]*0.05,segimg.shape[1]*0.95)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    if savefig:
        fig.savefig("plots/photom/{:s}_photom.png".format(objname))
        plt.close(fig)
    else:
        return fig, axes
