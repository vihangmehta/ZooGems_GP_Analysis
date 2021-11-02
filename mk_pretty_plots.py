from useful import *
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle, Circle
from extract_bc03 import TemplateSED_BC03
from matplotlib.colors import LogNorm, SymLogNorm
from astropy.convolution import convolve, Gaussian2DKernel, Tophat2DKernel, Box2DKernel
from gal_extinction import Gal_Extinction

import fsps
# sp = fsps.StellarPopulation(zcontinuous=1)

def getVminVmax(img,sigclip=False):

    size = img.shape[0]
    _img = img[int(np.floor(size / 3.0)) : int(np.ceil(size * 2.0 / 3.0)),
               int(np.floor(size / 3.0)) : int(np.ceil(size * 2.0 / 3.0))]
    _img = np.ma.masked_array(_img, mask=~np.isfinite(_img))
    if sigclip:
        _img = _img[(_img<np.ma.median(_img)+3*np.ma.std(_img)) & \
                    (_img>np.ma.median(_img)-3*np.ma.std(_img))]
    vmin = np.ma.median(_img) - 3.0*np.ma.std(_img)
    vmax = np.ma.median(_img) + 8.0*np.ma.std(_img)
    return vmin, vmax

def mkStampSheet(catalog):

    fig = plt.figure(figsize=(24,15),dpi=75)
    fig.subplots_adjust(left=0.005,right=0.995,bottom=0.01,top=0.99)
    ogs = fig.add_gridspec(5,2,hspace=0,wspace=0.02)

    cmin,cmax = -2,2
    kern = Gaussian2DKernel(fwhmToSigma(fwhm=2))
    # kern = Box2DKernel(1.5)
    kern2 = Tophat2DKernel(5)
    gal_ext = Gal_Extinction()

    for _ogs,galID in zip(ogs,sample):

        igs = _ogs.subgridspec(1,4,wspace=0,hspace=0)
        axes = igs.subplots()

        # Get ExpTime
        exp555 = getExpTime(galID,filt="F555W")
        exp850 = getExpTime(galID,filt="F850LP")

        # Readin stamps
        im555  = fitsio.getdata("fits/orig/{:s}_F555W_Stamp.fits".format(galID)) / exp555
        im850  = fitsio.getdata("fits/orig/{:s}_F850LP_Stamp.fits".format(galID)) / exp850
        im555c = fitsio.getdata("fits/{:s}_F555W_Stamp.fits".format(galID)) / exp555
        im850c = im850.copy()

        ### Galactic extinction correction
        entry = catalog[catalog["GAL_ID"]==galID][0]
        im555  /= gal_ext.calc_Alambda(filt="F555W", Av=entry["EXTINCT_AV"])[1]
        im850  /= gal_ext.calc_Alambda(filt="F850LP",Av=entry["EXTINCT_AV"])[1]
        im555c /= gal_ext.calc_Alambda(filt="F555W", Av=entry["EXTINCT_AV"])[1]
        im850c /= gal_ext.calc_Alambda(filt="F850LP",Av=entry["EXTINCT_AV"])[1]

        ### Stamp plotting
        vmin,vmax = getVminVmax(im555,sigclip=True)
        axes[0].pcolormesh(im555,cmap=plt.cm.Greys,norm=SymLogNorm(vmin=vmin/5,vmax=vmax*5,linthresh=np.abs(vmin/2),base=10))
        vmin,vmax = getVminVmax(im850,sigclip=True)
        axes[1].pcolormesh(im850,cmap=plt.cm.Greys,norm=SymLogNorm(vmin=vmin/5,vmax=vmax*5,linthresh=np.abs(vmin/2),base=10))

        ### Color stamp plotting
        im555c[(im555c<0)] = 0
        im850c[(im850c<0)] = 0
        im555c = convolve(im555c,kern,boundary="fill",fill_value=0)
        im850c = convolve(im850c,kern,boundary="fill",fill_value=0)
        color = (-2.5*np.log10(im555c)+zeropoint["F555W"]) - \
                (-2.5*np.log10(im850c)+zeropoint["F850LP"])

        SNthresh = 1.25 if galID!=sample[0] else 2.5

        mask555 = im555c / \
                  (fitsio.getdata("fits/orig/{:s}_F555W_Stamp.fits".format(galID),2)/exp555) > SNthresh
        mask850 = im850c / \
                  (fitsio.getdata("fits/orig/{:s}_F850LP_Stamp.fits".format(galID),2)/exp850) > SNthresh
        mask = np.ma.mask_or(mask555, mask850)
        color = np.ma.masked_array(color,mask=~mask)

        im = axes[2].pcolormesh(color,vmin=cmin,vmax=cmax,cmap=plt.cm.bwr)
        axes[2].set_facecolor([0.8,0.8,0.8])

        ### Plot contours
        segm  = fitsio.getdata("photom/smooth/{:s}_smooth_seg.fits".format(galID))
        stdev = np.std(im850[segm==0])
        levels = stdev*(np.array([1,3,5,10,30]))
        print(galID,levels[0],-2.5*np.log10(levels[0]/0.05/0.05)+zeropoint["F850LP"])

        axes[3].pcolormesh(color,vmin=cmin,vmax=cmax,cmap=plt.cm.bwr)
        axes[3].contour(im850c,levels=levels,colors='k',linewidths=0.9)
        axes[3].set_facecolor([0.8,0.8,0.8])

        ### Clump highlighting
        clumps = catalog[catalog["GAL_ID"]==galID]
        for clump in clumps:

            xy = (clump["CLUMP_X"]-1,clump["CLUMP_Y"]-1)

            if xy[0]<im555.shape[0]/2 and xy[1]<im555.shape[1]/2:
                xytext = xy[0]-im555.shape[0]/5, xy[1]-im555.shape[1]/5
            elif xy[0]>im555.shape[0]/2 and xy[1]<im555.shape[1]/2:
                xytext = xy[0]+im555.shape[0]/5, xy[1]-im555.shape[1]/5
            elif xy[0]<im555.shape[0]/2 and xy[1]>im555.shape[1]/2:
                xytext = xy[0]-im555.shape[0]/5, xy[1]+im555.shape[1]/5
            else:
                xytext = xy[0]+im555.shape[0]/5, xy[1]+im555.shape[1]/5

            axes[0].annotate("C%d"%clump["CLUMP_ID"],
                             xy=xy,xytext=xytext,arrowprops={"arrowstyle":"-","color":"red"},
                             color="red",fontsize=16,fontweight=600)
            axes[1].annotate("C%d"%clump["CLUMP_ID"],
                             xy=xy,xytext=xytext,arrowprops={"arrowstyle":"-","color":"red"},
                             color="red",fontsize=16,fontweight=600)
            axes[2].annotate("C%d"%clump["CLUMP_ID"],
                             xy=xy,xytext=xytext,arrowprops={"arrowstyle":"-","color":"k"},
                             color="k",fontsize=16,fontweight=600)

        axes[0].text(0.03,0.96,galID,color="k",fontsize=20,fontweight=600,va="top",ha="left",transform=axes[0].transAxes)

        ### Axes decorations
        for ax in axes:

            # if galID==sample[0]:
            #     ax.set_xlim(im555.shape[0]/10,im555.shape[0]*9/10)
            #     ax.set_ylim(im555.shape[1]/10,im555.shape[1]*9/10)
            # else:
            #     ax.set_xlim(im555.shape[0]/8,im555.shape[0]*7/8)
            #     ax.set_ylim(im555.shape[1]/8,im555.shape[1]*7/8)

            barx0,bary0,barh = ax.get_xlim()[1]*0.98, np.diff(ax.get_ylim())*0.05, np.diff(ax.get_ylim())*0.01
            ax.add_patch(Rectangle(xy=(barx0-1/FITS_pixscale,bary0),width=1/FITS_pixscale,height=barh,facecolor="k",edgecolor='none',lw=0,alpha=0.9))
            # ax.text(barx0-1/FITS_pixscale/2, bary0+1.1*barh, "1\"", va="top", ha="center", fontsize=13, color="k")
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.set_aspect(1)

    cbax = fig.add_axes([0.6,0.12,0.3,0.03])
    cbax = fig.colorbar(mappable=im, cax=cbax, orientation="horizontal")
    cbax.set_label("B$_{555}$ - I$_{850}$",fontsize=32)
    [tick.set_fontsize(22) for tick in cbax.ax.get_xticklabels()]

    fig.savefig("plots/GPZoo_stampsheet.png",dpi=150)

def mkTemplatesColors(ages,metal,sfh,tau=None,emlines=True):

    template = TemplateSED_BC03(metallicity=metal, age=ages, sfh=sfh, tau=tau, Av=0,
                                dust='calzetti', emlines=emlines,
                                redshift=None, igm=False,
                                imf='chab', res='hr', units='flambda',
                                rootdir='/data/highzgal/mehta/Software/galaxev12/',workdir='.',
                                library_version=2012,cleanup=True,verbose=False)
    template.generate_sed()

    mag_f555w,mag_f850lp = np.zeros((2,len(ages)))
    for i,age in enumerate(ages):
        mag_f555w[i]  = -2.5*np.log10(calc_filter_flux(template.sed["waves"],template.sed["spec%d"%(i+1)],filt="f555w"))
        mag_f850lp[i] = -2.5*np.log10(calc_filter_flux(template.sed["waves"],template.sed["spec%d"%(i+1)],filt="f850lp"))

    return mag_f555w - mag_f850lp

def mkTemplatesColorsFSPS(ages,zsol,sfh,zred=0.25,emlines=True,new=False):

    if emlines: savename = "fsps/template_%s_Z%.1f.fits"%(sfh,zsol)
    else:       savename = "fsps/template_%s_Z%.1f_nolines.fits"%(sfh,zsol)

    if new:

        sp.params["zred"] = zred
        sp.params["logzsol"] = np.log10(zsol)
        sp.params["add_neb_emission"] = emlines
        sp.params["add_neb_continuum"] = emlines

        if sfh=="ssp":
            sp.params["sfh"] = 0
        elif sfh=="constant":
            sp.params["sfh"] = 1
            sp.params["const"] = 1

        dtype = [("waves",float)]
        for i,age in enumerate(ages): dtype += [("spec%d"%(i+1),float)]
        template = np.recarray(len(sp.wavelengths),dtype=dtype)
        for x in template.dtype.names: template[x] = -99

        template["waves"] = sp.wavelengths * (1+zred)
        for i,age in enumerate(ages):
            print("\rGenerating spectrum #%d/%d (%.3f) ... "%(i+1,len(ages),age),end="",flush=True)
            template["spec%d"%(i+1)] = sp.get_spectrum(tage=age,peraa=True)[1]
            template["spec%d"%(i+1)] /= (1+zred)
        print("done.")

        fitsio.writeto(savename,template,overwrite=True)

    template = fitsio.getdata(savename)

    mag_f555w,mag_f850lp = np.zeros((2,len(ages)))
    for i,age in enumerate(ages):
        mag_f555w[i]  = -2.5*np.log10(calc_filter_flux(template["waves"],template["spec%d"%(i+1)],filt="f555w"))
        mag_f850lp[i] = -2.5*np.log10(calc_filter_flux(template["waves"],template["spec%d"%(i+1)],filt="f850lp"))

    return mag_f555w - mag_f850lp

def mkColorPlot(catalog):

    colorClump  = catalog["CLUMP_ORGMAG_F555W"] - catalog["CLUMP_ORGMAG_F850LP"]
    colorClump2 = catalog["CLUMP_NEWMAG_F555W"] - catalog["CLUMP_NEWMAG_F850LP"]
    # colorGalaxy = catalog["GAL_MAG_AUTO_F555W"] - catalog["GAL_MAG_AUTO_F850LP"]

    colorGalaxy = np.zeros(len(catalog))
    for i,entry in enumerate(catalog):
        idx = np.where(catalog["GAL_ID"]==entry["GAL_ID"])[0]
        clumpFluxF555W  = np.sum(catalog["CLUMP_NEWFLUX_F555W" ][idx])
        clumpFluxF850LP = np.sum(catalog["CLUMP_NEWFLUX_F850LP"][idx])
        galFluxF555W    = entry["GAL_FLUX_AUTO_F555W" ]
        galFluxF850LP   = entry["GAL_FLUX_AUTO_F850LP"]
        colorGalaxy[i]  = -2.5*np.log10(galFluxF555W-clumpFluxF555W) + 2.5*np.log10(galFluxF850LP-clumpFluxF850LP)

    fig,ax = plt.subplots(1,1,figsize=(10,9.5),dpi=75,tight_layout=True)

    markers = ['o','v','s','p','*','D','P','X','<']
    colors  = plt.cm.tab10(np.arange(0,1.001,0.1))

    for objname,color,marker in zip(sample,colors,markers):
        idx = np.where(catalog["GAL_ID"]==objname)[0]
        ax.plot(colorGalaxy[idx],colorClump[idx],
                markerfacecolor="none",markeredgecolor=color,marker=marker,markersize=15,lw=2,ls='--',mew=2)
        ax.plot(colorGalaxy[idx],colorClump2[idx],
                color=color,marker=marker,markersize=15,lw=2,label=objname)

    ax.plot([-5,5],[-5,5],color='k',lw=1,alpha=0.9)
    ax.set_aspect(1)
    ax.set_xlim(-0.43,0.6)
    ax.set_ylim(-0.43,0.6)
    ax.set_xlabel("$(B_{555} - I_{850})_{galaxy}$",fontsize=20)
    ax.set_ylabel("$(B_{555} - I_{850})_{clump}$",fontsize=20)
    ax.legend(loc="best",fontsize=20)

    ax.grid(linestyle=':', linewidth=0.5, color='k', which="both")
    [tick.set_fontsize(15) for tick in ax.get_xticklabels()+ax.get_yticklabels()]

    fig.savefig("plots/GPZoo_colors.png")

def mkSPSColorPlot(catalog):

    colorClump  = catalog["CLUMP_ORGMAG_F555W"] - catalog["CLUMP_ORGMAG_F850LP"]
    colorClump2 = catalog["CLUMP_NEWMAG_F555W"] - catalog["CLUMP_NEWMAG_F850LP"]
    # colorGalaxy = catalog["GAL_MAG_AUTO_F555W"] - catalog["GAL_MAG_AUTO_F850LP"]

    colorGalaxy = np.zeros(len(catalog))
    for i,entry in enumerate(catalog):
        idx = np.where(catalog["GAL_ID"]==entry["GAL_ID"])[0]
        clumpFluxF555W  = np.sum(catalog["CLUMP_NEWFLUX_F555W" ][idx])
        clumpFluxF850LP = np.sum(catalog["CLUMP_NEWFLUX_F850LP"][idx])
        galFluxF555W    = entry["GAL_FLUX_AUTO_F555W" ]
        galFluxF850LP   = entry["GAL_FLUX_AUTO_F850LP"]
        colorGalaxy[i]  = -2.5*np.log10(galFluxF555W-clumpFluxF555W) + 2.5*np.log10(galFluxF850LP-clumpFluxF850LP)
        # print(entry["GAL_ID"],entry["GAL_MAG_AUTO_F555W"],-2.5*np.log10(galFluxF555W-clumpFluxF555W)+23.9,entry["GAL_MAG_AUTO_F850LP"],-2.5*np.log10(galFluxF850LP-clumpFluxF850LP)+23.9)

    iuniq = np.unique(catalog["GAL_ID"],return_index=True)[1]
    colorGalaxy = colorGalaxy[iuniq]

    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(10,12),dpi=75,sharex=True,gridspec_kw={"height_ratios":[1,3]})
    fig.subplots_adjust(left=0.08,right=0.98,bottom=0.08,top=0.98,hspace=0.05)

    bins = np.arange(-5,5,0.05)
    ax1.hist(colorClump2,bins=bins,color="tab:blue",lw=0,alpha=0.8,label="$(B_{555} - I_{850})_{SC}$")
    # ax1.hist(colorClump, bins=bins,color="darkblue",lw=3,ls="--",alpha=0.6,histtype="step",label="$(B_{555} - I_{850})_{ObsC}$")
    ax1.hist(colorGalaxy,bins=bins,color="tab:red",lw=0,alpha=0.8,label="$(B_{555} - I_{850})_{G}$")
    ax1.set_ylim(0,5)

    ages = np.arange(6,10.13,0.02)
    ax2.plot(mkTemplatesColorsFSPS(ages=10**(ages-9),zsol=0.2,sfh="constant",emlines=True),ages,color='tab:red', ls='-',lw=4,alpha=0.9)
    ax2.plot(mkTemplatesColorsFSPS(ages=10**(ages-9),zsol=0.4,sfh="constant",emlines=True),ages,color='tab:red', ls='-',lw=2,alpha=0.7)
    ax2.plot(mkTemplatesColorsFSPS(ages=10**(ages-9),zsol=0.2,sfh="ssp",emlines=True),     ages,color='tab:blue',ls='-',lw=4,alpha=0.9)
    ax2.plot(mkTemplatesColorsFSPS(ages=10**(ages-9),zsol=0.4,sfh="ssp",emlines=True),     ages,color='tab:blue',ls='-',lw=2,alpha=0.7)

    ax2.plot(mkTemplatesColorsFSPS(ages=10**(ages-9),zsol=0.2,sfh="constant",emlines=False),ages,color='tab:red', ls='--',lw=4,alpha=0.9)
    ax2.plot(mkTemplatesColorsFSPS(ages=10**(ages-9),zsol=0.4,sfh="constant",emlines=False),ages,color='tab:red', ls='--',lw=2,alpha=0.7)
    ax2.plot(mkTemplatesColorsFSPS(ages=10**(ages-9),zsol=0.2,sfh="ssp",emlines=False),     ages,color='tab:blue',ls='--',lw=4,alpha=0.9)
    ax2.plot(mkTemplatesColorsFSPS(ages=10**(ages-9),zsol=0.4,sfh="ssp",emlines=False),     ages,color='tab:blue',ls='--',lw=2,alpha=0.7)

    ax2.add_patch(Rectangle((-99,-99),0,0,color='tab:red',lw=0,label="CSF"))
    ax2.add_patch(Rectangle((-99,-99),0,0,color='tab:blue',lw=0,label="SSP"))
    ax2.plot(-99,-99,c='k',lw=3,ls='-',label="with neb. contrib.")
    ax2.plot(-99,-99,c='k',lw=3,ls='--',label="w/o neb. contrib.")
    ax2.plot(-99,-99,c='k',lw=4,ls='-',alpha=0.9,label="$0.2 Z_\\odot$")
    ax2.plot(-99,-99,c='k',lw=2,ls='-',alpha=0.7,label="$0.4 Z_\\odot$")

    ax1.set_ylabel("N",fontsize=20)
    ax2.set_ylabel("log Stellar Age [yr]",fontsize=20)
    ax2.set_xlabel("$(B_{555} - I_{850})$",fontsize=20)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax2.set_xlim(-0.43,0.6)
    ax2.set_ylim(6.4,10.15)

    ax1.legend(loc=2,fontsize=20)
    # ax2.legend(loc="best",fontsize=20)

    handles, labels = ax2.get_legend_handles_labels()
    isort = [4,5,0,1,2,3]
    ax2.legend([handles[idx] for idx in isort],
               [ labels[idx] for idx in isort],loc="best",fontsize=20)

    for ax in [ax1,ax2]:
        ax.grid(linestyle=':', linewidth=0.5, color='k', which="both")
        [tick.set_fontsize(15) for tick in ax.get_xticklabels()+ax.get_yticklabels()]

    fig.savefig("plots/GPZoo_colors_sps.png")

def mkNebContribColorPlot():

    ages = np.arange(6,10.13,0.02)
    spec_emlines = fitsio.getdata("fsps/template_constant_Z0.2.fits")
    spec_nolines = fitsio.getdata("fsps/template_constant_Z0.2_nolines.fits")
    mag_f555w,mag_f850lp = np.zeros((2,len(ages)))
    for i,age in enumerate(ages):
        spec_emlines["spec%d"%(i+1)] = spec_emlines["spec%d"%(i+1)] - spec_nolines["spec%d"%(i+1)]
        mag_f555w[i]  = -2.5*np.log10(calc_filter_flux(spec_emlines["waves"],spec_emlines["spec%d"%(i+1)],filt="f555w"))
        mag_f850lp[i] = -2.5*np.log10(calc_filter_flux(spec_emlines["waves"],spec_emlines["spec%d"%(i+1)],filt="f850lp"))
    print((mag_f555w - mag_f850lp)[-10:])

    fig,ax = plt.subplots(1,1,figsize=(10,8),dpi=75,tight_layout=True)

    ax.plot(mag_f555w - mag_f850lp, ages, color='tab:green',ls='-',lw=2,alpha=0.9)
    ax.set_ylabel("log Stellar Age [yr]",fontsize=20)
    ax.set_xlabel("$(V-I)$",fontsize=20)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_xlim(0.7,0.82)
    ax.set_ylim(6.4,10.15)
    ax.grid(linestyle=':', linewidth=0.5, color='k', which="both")
    [tick.set_fontsize(15) for tick in ax.get_xticklabels()+ax.get_yticklabels()]

def mkTemplateSpectra(age,label,zsol):

    sp = fsps.StellarPopulation(zcontinuous=1)

    sp.params["sfh"] = 0
    sp.params["logzsol"] = np.log10(zsol)
    wave = sp.wavelengths

    sp.params["add_neb_emission"] = True
    sp.params["add_neb_continuum"] = True
    spec_emline = sp.get_spectrum(tage=age)[1]

    sp.params["add_neb_emission"] = False
    sp.params["add_neb_continuum"] = False
    spec_noline = sp.get_spectrum(tage=age)[1]

    np.savetxt("fsps/template_ssp_%s_Zsol%.1f.txt"%(label,zsol),
                np.vstack([wave,spec_emline,spec_noline]).T,
                fmt="%15.8e%15.8e%15.8e",
                header="%13s%15s%15s"%("wave","Fnu (w/ neb)","Fnu (no neb)"),comments="#")

def mkTemplatePlot():

    # mkTemplateSpectra(age=0.01,label="10Myr", zsol=0.4)
    # mkTemplateSpectra(age=0.012,label="12Myr",zsol=0.4)
    # mkTemplateSpectra(age=1,   label="1Gyr",  zsol=0.4)

    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(15,10),dpi=75,sharex=True)
    fig.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.98,hspace=0.03)

    wave,spec_emlines,spec_nolines = np.genfromtxt("fsps/template_ssp_12Myr_Zsol0.4.txt",unpack=True)
    idx = np.argmin(np.abs(wave-5500))
    ax1.plot(wave/1e4,spec_nolines/spec_nolines[idx],color='k',lw=1.2,alpha=0.8,label="12Myr SSP")
    ax1.plot(wave/1e4,spec_emlines/spec_emlines[idx],color='tab:blue',lw=1.2,alpha=0.8,label="12Myr SSP + neb")
    ax1.plot(wave/1e4,spec_emlines/spec_emlines[idx]-
                      spec_nolines/spec_nolines[idx],color='green',lw=1.2,alpha=0.8,label="Neb. contrib.")

    wave,spec_emlines,spec_nolines = np.genfromtxt("fsps/template_ssp_1Gyr_Zsol0.4.txt",unpack=True)
    idx = np.argmin(np.abs(wave-5500))
    ax2.plot(wave/1e4,spec_nolines/spec_nolines[idx],color='k',lw=1.2,alpha=0.8,label="1Gyr SSP")
    ax2.plot(wave/1e4,spec_emlines/spec_emlines[idx],color='tab:red',lw=1.2,alpha=0.8,label="1Gyr SSP + neb")
    ax2.plot(wave/1e4,spec_emlines/spec_emlines[idx]-
                      spec_nolines/spec_nolines[idx],color='green',lw=1.2,alpha=0.8,label="Neb. contrib.")

    ax1.set_ylim(1e-4,3e1)
    ax2.set_ylim(1e-2,3e0)
    ax2.set_xlabel("Rest-frame Wavelength [$\\mu$m]",fontsize=20)

    for ax in [ax1,ax2]:
        ax.legend(fontsize=20)
        ax.set_xlim(0.2,1)
        ax.set_yscale("log")
        ax.set_ylabel("$F_\\nu$ [erg/s/cm$^2$/Hz]",fontsize=20)
        ax.grid(linestyle=':', linewidth=0.5, color='k', which="both")
        [tick.set_fontsize(15) for tick in ax.get_xticklabels()+ax.get_yticklabels()]

def mkFracLightPlot(catalog):

    fig,ax = plt.subplots(1,1,figsize=(10,6),dpi=75,tight_layout=True)

    clumpFluxMax,clumpFlux,galFlux = {},{},{}
    for filt in ["F555W","F850LP"]:
        clumpFluxMax[filt],clumpFlux[filt],galFlux[filt] = np.zeros((3,len(sample)))

    for i,galID in enumerate(sample):
        idx = np.where(catalog["GAL_ID"]==galID)[0]
        for filt in ["F555W","F850LP"]:
            clumpFluxMax[filt][i] = np.max(catalog["CLUMP_NEWFLUX_{:s}".format(filt)][idx])
            clumpFlux[filt][i]    = np.sum(catalog["CLUMP_NEWFLUX_{:s}".format(filt)][idx])
            galFlux[filt][i]      = catalog["GAL_FLUX_AUTO_{:s}".format(filt)][idx][0]

    ax.hist(clumpFlux["F555W"]/galFlux["F555W"],bins=np.arange(0,1,0.1),color='tab:blue',alpha=0.7,label="F555W")
    ax.hist(clumpFlux["F850LP"]/galFlux["F850LP"],bins=np.arange(0,1,0.1),color='tab:red',alpha=0.7,label="F850LP")

    # ax.hist(clumpFluxMax["F555W"]/galFlux["F555W"],bins=np.arange(0,1,0.1),color='tab:blue',alpha=0.9,histtype="step",lw=3,ls='--')
    # ax.hist(clumpFluxMax["F850LP"]/galFlux["F850LP"],bins=np.arange(0,1,0.1),color='tab:red',alpha=0.9,histtype="step",lw=3,ls='--')

    ax.legend(fontsize=20)
    ax.set_xlim(-0.05,1.05)
    ax.set_ylim(0,5)
    ax.set_xlabel("$f_{clump}/f_{galaxy}$",fontsize=20)
    [tick.set_fontsize(18) for tick in ax.get_xticklabels()+ax.get_yticklabels()]

if __name__ == '__main__':

    catalog = fitsio.getdata("catalogs/GPZoo_clump_catalog.fits")

    mkStampSheet(catalog=catalog)

    mkColorPlot(catalog=catalog)
    mkSPSColorPlot(catalog=catalog)
    mkFracLightPlot(catalog=catalog)

    mkNebContribColorPlot()
    mkTemplatePlot()

    plt.show()
