from useful import *
import numpy.lib.recfunctions as rfn

def getDType():

    dtype = [("GAL_ID","U10"),
             ("CLUMP_ID",int),("CLUMP_X",float),("CLUMP_Y",float),("CLUMP_RA",float),("CLUMP_DEC",float),
             ("GAL_XC",float),("GAL_YC",float),("GAL_RA",float),("GAL_DEC",float),
             ("GAL_REFF",float),("GAL_SMA",float),("GAL_SMB",float),("GAL_THETA",float),
             ("GAL_REFF_XY",float),("GAL_SMA_XY",float),("GAL_SMB_XY",float),
             ("CLUMP_DISTANCE_XY",float),("CLUMP_DISTANCE",float),("CLUMP_DISTNORM",float),("CLUMP_DIST_SMA",float),
             ("PSF_WIDTH_AVG",float)]

    dtype += [("NPIX_APER", float), ("EXTINCT_EBV", float),("EXTINCT_AV", float)]

    for filt in filterset:
        dtype.extend([("EXTINCT_{:s}".format(filt), float),
                      ("GAL_FLUX_AUTO_{:s}".format(filt), float),
                      ("GAL_FLUXERR_AUTO_{:s}".format(filt), float),
                      ("GAL_MAG_AUTO_{:s}".format(filt), float),
                      ("GAL_MAGERR_AUTO_{:s}".format(filt), float),
                      ("GAL_FLUX_ISO_{:s}".format(filt), float),
                      ("GAL_FLUXERR_ISO_{:s}".format(filt), float),
                      ("GAL_MAG_ISO_{:s}".format(filt), float),
                      ("GAL_MAGERR_ISO_{:s}".format(filt), float),
                      ("CLUMP_FLUX_{:s}".format(filt), float),
                      ("CLUMP_FLUXERR_{:s}".format(filt), float),
                      ("CLUMP_MAG_{:s}".format(filt), float),
                      ("CLUMP_MAGERR_{:s}".format(filt), float),
                      ("CLUMP_ORGFLUX_{:s}".format(filt), float),
                      ("CLUMP_ORGFLUXERR_{:s}".format(filt), float),
                      ("CLUMP_ORGMAG_{:s}".format(filt), float),
                      ("CLUMP_ORGMAGERR_{:s}".format(filt), float),
                      ("CLUMP_UNDFLUX_{:s}".format(filt), float),
                      ("CLUMP_UNDFLUXERR_{:s}".format(filt), float),
                      ("CLUMP_UNDMAG_{:s}".format(filt), float),
                      ("CLUMP_UNDMAGERR_{:s}".format(filt), float),
                      ("CLUMP_DIFFLUX_{:s}".format(filt), float),
                      ("CLUMP_DIFFSTD_{:s}".format(filt), float),
                      ("CLUMP_IMGBKGD_{:s}".format(filt), float),
                      ("CLUMP_NEWFLUX_{:s}".format(filt), float),
                      ("CLUMP_NEWFLUXERR_{:s}".format(filt), float),
                      ("CLUMP_NEWMAG_{:s}".format(filt), float),
                      ("CLUMP_NEWMAGERR_{:s}".format(filt), float),
                      ("CLUMP_GALFLUXAVG_{:s}".format(filt), float),
                      ("CLUMP_GALFLUXAVGERR_{:s}".format(filt), float)])
    return dtype

def mkClumpCatalog(sample,savename,destdir="boxcar8"):

    N = getTotalNClumps(sample=sample,destdir=destdir)
    catalog = np.recarray(N,dtype=getDType())
    galcat = fitsio.getdata("photom/galaxy/galaxy_photom.fits")

    i = 0
    for j,objname in enumerate(sample):

        print("\rProcessing {0:s} ({1:d}/{2:d}) ... ".format(objname,j+1,len(sample)),end="")
        photname = "photom/{1:s}/{0:s}_phot.fits".format(objname,destdir)

        try:
            phot = fitsio.getdata(photname)
            for x in phot.dtype.names:
                if x in catalog.dtype.names:
                    catalog[i:i+len(phot)][x] = phot[x]
                elif "CLUMP_"+x in catalog.dtype.names and x!="ID":
                    catalog[i:i+len(phot)]["CLUMP_"+x] = phot[x]

            catalog[i:i+len(phot)]["CLUMP_ID"] = phot["ID"]

            for x in galcat.dtype.names:
                if x in catalog.dtype.names:
                    catalog[i:i+len(phot)][x] = galcat[x][j]
                elif "GAL_"+x in catalog.dtype.names:
                    catalog[i:i+len(phot)]["GAL_"+x] = galcat[x][j]

            catalog[i:i+len(phot)]["GAL_ID"] = objname

            i += len(phot)

        except OSError:
            pass

    print("done!")

    catalog = rfn.append_fields(catalog,data=catalog["CLUMP_ID"],names="CLUMP_ORGID",dtypes=int,usemask=False,asrecarray=True)
    for galid in sample:
        idx = np.where(catalog["GAL_ID"]==galid)[0]
        newIDs = np.arange(len(idx))+1
        isort = np.argsort(catalog["CLUMP_MAG_F850LP"][idx])
        for x in catalog.dtype.names:
            catalog[x][idx] = catalog[x][idx[isort]]
        catalog["CLUMP_ID"][idx] = newIDs

    print("Final catalog: {:d} clumps".format(len(catalog)))
    fitsio.writeto(savename, catalog, overwrite=True)

if __name__ == '__main__':

    mkClumpCatalog(sample=sample,savename="catalogs/GPZoo_clump_catalog.fits")
