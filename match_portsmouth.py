from useful import *
import esutil

def matchRADec(ra1, dec1, ra2, dec2, crit, maxmatch=0):
    """
    Matches two catalogs by (ra,dec)
    """
    h = esutil.htm.HTM(10)
    crit = crit / 3600.0  # crit arcsec --> deg
    m1, m2, d12 = h.match(ra1, dec1, ra2, dec2, crit, maxmatch=maxmatch)
    return m1, m2, d12

def main():

    coords = SkyCoord([sexaRaDecDict[x] for x in sample],frame="icrs")
    portsmouth_dr8  = fitsio.getdata("/data/highzgal/mehta/SDSS/portsmouth_stellarmass_starforming_salp-26.fits")
    portsmouth_dr12 = fitsio.getdata("/data/highzgal/mehta/SDSS/portsmouth_stellarmass_starforming_salp-DR12-boss.fits")

    dtype = []
    for x in portsmouth_dr8.dtype.descr:
        if x[0] not in [_[0] for _ in dtype]: dtype.append(x)
    for x in portsmouth_dr12.dtype.descr:
        if x[0] not in [_[0] for _ in dtype]: dtype.append(x)

    matched = np.recarray(len(sample),dtype=[("ObjID","U10"),("ObjRA",float),("ObjDec",float),("ObjDist",float)]+dtype)
    for x in matched.dtype.names: matched[x] = -99
    matched["ObjID"]       = sample
    matched["ObjRA"]       = coords.ra.deg
    matched["ObjDec"]      = coords.dec.deg

    m1,m2,d12 = matchRADec(coords.ra.deg,coords.dec.deg,portsmouth_dr8["RA"],portsmouth_dr8["Dec"],crit=3,maxmatch=1)
    matched["ObjDist"][m1] = d12 * 3600
    for x in portsmouth_dr8.dtype.names: matched[x][m1] = portsmouth_dr8[x][m2]

    m1,m2,d12 = matchRADec(coords.ra.deg,coords.dec.deg,portsmouth_dr12["RA"],portsmouth_dr12["Dec"],crit=3,maxmatch=1)
    matched["ObjDist"][m1] = d12 * 3600
    for x in portsmouth_dr12.dtype.names: matched[x][m1] = portsmouth_dr12[x][m2]

    fitsio.writeto("catalogs/GPZoo_catalog.portsmouth_stellarmass.fits",matched,overwrite=True)

if __name__ == '__main__':

    main()
