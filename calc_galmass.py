from useful import *
from astropy.cosmology import Planck15

import fsps
# sp = fsps.StellarPopulation(zcontinuous=1)

ages = np.arange(6,10.13,0.02)

def mkTemplatesFSPS(ages,zsol,sfh,zred=0.25,emlines=True,new=False):

    if emlines: savename = "fsps/template_%s_Z%.1f.fits"%(sfh,zsol)
    else:       savename = "fsps/template_%s_Z%.1f_nolines.fits"%(sfh,zsol)

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

    masses = np.recarray(len(ages),dtype=[("age",float),("mass",float)])
    masses["age"] = ages

    template["waves"] = sp.wavelengths * (1+zred)
    for i,age in enumerate(ages):
        print("\rGenerating spectrum #%d/%d (%.3f) ... "%(i+1,len(ages),age),end="",flush=True)
        template["spec%d"%(i+1)] = sp.get_spectrum(tage=age,peraa=True)[1]
        template["spec%d"%(i+1)] /= (1+zred)
        masses["mass"][i] = sp.stellar_mass
    print("done.")

    hdu = fitsio.HDUList([fitsio.PrimaryHDU(),
                          fitsio.BinTableHDU.from_columns(template),
                          fitsio.BinTableHDU.from_columns(masses)])
    hdu.writeto(savename,overwrite=True)

def calcTemplateMags(template):

    mag_f555w,mag_f850lp = np.zeros((2,len(ages)))
    for i,age in enumerate(ages):
        mag_f555w[i]  = -2.5*np.log10(calc_filter_flux(template["waves"],template["spec%d"%(i+1)],filt="f555w"))
        mag_f850lp[i] = -2.5*np.log10(calc_filter_flux(template["waves"],template["spec%d"%(i+1)],filt="f850lp"))

    return mag_f555w, mag_f850lp, mag_f555w - mag_f850lp

def setupTemplates():

    mkTemplatesFSPS(ages=10**(ages-9),zsol=0.2,sfh="constant",emlines=True,new=True)
    mkTemplatesFSPS(ages=10**(ages-9),zsol=0.4,sfh="constant",emlines=True,new=True)
    mkTemplatesFSPS(ages=10**(ages-9),zsol=0.2,sfh="ssp",emlines=True,new=True)
    mkTemplatesFSPS(ages=10**(ages-9),zsol=0.4,sfh="ssp",emlines=True,new=True)
    mkTemplatesFSPS(ages=10**(ages-9),zsol=0.2,sfh="constant",emlines=False,new=True)
    mkTemplatesFSPS(ages=10**(ages-9),zsol=0.4,sfh="constant",emlines=False,new=True)
    mkTemplatesFSPS(ages=10**(ages-9),zsol=0.2,sfh="ssp",emlines=False,new=True)
    mkTemplatesFSPS(ages=10**(ages-9),zsol=0.4,sfh="ssp",emlines=False,new=True)

def calc_mass(app_mag,spec_mag,spec_mass,zred=0.25):

    Lsun = 4e33
    spec_lum  = 10**(spec_mag/-2.5)
    spec_flux = spec_lum * Lsun / (4*np.pi*Planck15.luminosity_distance(z=zred).cgs.value**2) * (1+zred)

    flux = 10**((app_mag+48.6)/-2.5)

    mass = spec_mass * flux / spec_flux
    return np.log10(mass)

def main(catalog,masscat):

    clumpMagF555W  = catalog["CLUMP_NEWMAG_F555W"]
    clumpMagF850LP = catalog["CLUMP_NEWMAG_F850LP"]
    colorClump = clumpMagF555W - clumpMagF850LP

    galMagF555W,galMagF850LP,colorGalaxy = np.zeros((3,len(catalog)))
    for i,entry in enumerate(catalog):
        idx = np.where(catalog["GAL_ID"]==entry["GAL_ID"])[0]
        clumpFluxF555W  = np.sum(catalog["CLUMP_NEWFLUX_F555W" ][idx])
        clumpFluxF850LP = np.sum(catalog["CLUMP_NEWFLUX_F850LP"][idx])
        galFluxF555W    = entry["GAL_FLUX_AUTO_F555W" ]
        galFluxF850LP   = entry["GAL_FLUX_AUTO_F850LP"]
        galMagF555W[i]  = -2.5*np.log10(galFluxF555W-clumpFluxF555W) + 23.9
        galMagF850LP[i] = -2.5*np.log10(galFluxF850LP-clumpFluxF850LP) + 23.9

    iuniq = np.unique(catalog["GAL_ID"],return_index=True)[1]
    catalog, galMagF555W, galMagF850LP = catalog[iuniq], galMagF555W[iuniq], galMagF850LP[iuniq]
    colorGalaxy  = galMagF555W - galMagF850LP

    templates = ["fsps/template_constant_Z0.2.fits",
                 "fsps/template_constant_Z0.2_nolines.fits",
                 "fsps/template_constant_Z0.4.fits",
                 "fsps/template_constant_Z0.4_nolines.fits"]

    newages  = np.zeros((len(catalog),len(templates)))
    newmass  = np.zeros((len(catalog),len(templates)))
    sdssmass = np.zeros(len(catalog))

    for k,savename in enumerate(templates):

        template = fitsio.getdata(savename,1)
        masses   = fitsio.getdata(savename,2)
        spec_m555,spec_m850,spec_color = calcTemplateMags(template)

        iageMax = np.where(masses["age"]==max(masses["age"][masses["age"]<=Planck15.age(z=0.25).value]))[0][0]
        iageMin = np.where(masses["age"]==max(masses["age"][masses["age"]<=10**(7-9)]))[0][0]
        spec_color[:iageMin] = -99

        for j,(entry,m555Gal,m850Gal,colGal) in enumerate(zip(catalog,galMagF555W,galMagF850LP,colorGalaxy)):

            iage = np.argmin(np.abs(colGal - spec_color))
            ageGal = masses["age"][min(iage,iageMax)]
            mass555Gal = calc_mass(app_mag=m555Gal,spec_mag=spec_m555[iage],spec_mass=masses["mass"][iage])
            mass850Gal = calc_mass(app_mag=m850Gal,spec_mag=spec_m850[iage],spec_mass=masses["mass"][iage])
            massGal = np.log10((10**mass555Gal+10**mass850Gal)/2)

            entry2 = masscat[masscat["ObjID"]==entry["GAL_ID"]][0]

            newages[j,k] = ageGal
            newmass[j,k] = massGal
            sdssmass[j]  = entry2["LOGMASS"]

    print("Template name: {:s}".format(savename))
    print("{:^12s}||{:^64s}||{:^8s}".format("","Galaxy","SDSS"))
    print("{:^12s}||{:^8s}{:^8s}|{:^24s}|{:^22s}||{:^8s}".format("ID","m555","m850","age","mass","mass"))
    print(''.join(['-']*88))
    for j,entry in enumerate(catalog):
        print("{:^12s}||{:^8.2f}{:^8.2f}|{:^8.3f}[{:>6.3f},{:>6.3f}] |{:^8.2f}[{:>5.2f},{:>5.2f}] ||{:^8.2f}".format(
                        entry["GAL_ID"],galMagF555W[j],galMagF850LP[j],
                        np.median(newages[j,:]),np.min(newages[j,:]),np.max(newages[j,:]),
                        np.median(newmass[j,:]),np.min(newmass[j,:]),np.max(newmass[j,:]),
                        sdssmass[j]))

if __name__ == '__main__':

    catalog = fitsio.getdata("catalogs/GPZoo_clump_catalog.fits")
    masscat = fitsio.getdata("catalogs/GPZoo_catalog.portsmouth_stellarmass.fits")

    # setupTemplates()
    main(catalog,masscat)
