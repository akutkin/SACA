import os


def clean_difmap(fname, stokes, mapsize_clean, path=None, path_to_script=None,
                 mapsize_restore=None, outfname=None, outpath=None):
    """
    Map self-calibrated uv-data in difmap.
    :param fname:
        Path to file.
    :param stokes:
        Stokes parameter 'i', 'q', 'u' or 'v'.
    :param mapsize_clean:
        Tuple (mapsize_clean, pixsize [mas])

    """
    if path is None:
        path = os.getcwd()
    elif not path.endswith("/"):
        path = path + "/"
    if outpath is None:
        outpath = os.getcwd()
    elif not outpath.endswith("/"):
        outpath = outpath + "/"

    difmapout = open("difmap_commands", "w")
    difmapout.write("observe " + path + fname + "\n")
    difmapout.write("mapsize " + str(mapsize_clean[0]) + ', ' +
                    str(mapsize_clean[1]) + "\n")
    difmapout.write("@" + path_to_script + " " + stokes)
    difmapout.write("mapsize " + str(mapsize_restore[0]) + ', ' +
                    str(mapsize_restore[1]) + "\n")
    if outpath is None:
        outpath = path
    elif not outpath.endswith("/"):
        outpath = outpath + "/"
    difmapout.write("wmap " + outpath + outfname)

# DIFMAP_MAPPSR
def difmap_mappsr(source, isll, centre_ra_deg, centre_dec_deg, uvweightstr,
                  experiment, difmappath, uvprefix, uvsuffix, jmsuffix, \
                  saveagain):
    difmapout = open("difmap_commands", "w")
    difmapout.write("float pkflux\n")
    difmapout.write("float peakx\n")
    difmapout.write("float peaky\n")
    difmapout.write("float finepeakx\n")
    difmapout.write("float finepeaky\n")
    difmapout.write("float rmsflux\n")
    difmapout.write("integer ilevs\n")
    difmapout.write("float lowlev\n")
    difmapout.write("obs " + sourceuvfile + "\n")
    difmapout.write("mapsize 1024," + str(pixsize) + "\n")
    difmapout.write("select ll,1,2,3,4\n")
    difmapout.write("invert\n");
    difmapout.write("wtscale " + str(threshold) + "\n")
    difmapout.write("obs " + sourceuvfile + "\n")
    if reweight:
        difmapout.write("uvaver 180,true\n")
    else:
        difmapout.write("uvaver 20\n")
    difmapout.write("mapcolor none\n")
    difmapout.write("uvweight " + uvweightstr + "\n")

    #Find the peak from the combined first
    if isll:
        difmapout.write("select ll,1,2,3,4\n")
    else:
        difmapout.write("select i,1,2\n")
    if experiment == 'v190k':
        difmapout.write("select rr,1,2,3,4\n")
    difmapout.write("peakx = peak(x,max)\n")
    difmapout.write("peaky = peak(y,max)\n")
    difmapout.write("shift -peakx,-peaky\n")
    difmapout.write("mapsize 1024,0.1\n")
    difmapout.write("finepeakx = peak(x,max)\n")
    difmapout.write("finepeaky = peak(y,max)\n")

    #Do each individual band, one at a time
    maxif = 4
    pols = ['rr','ll']
    for i in range(maxif):
        for p in pols:
            difmapout.write("select " + p + "," + str(i+1) + "\n")
            write_difmappsrscript(source, p + "." + str(i+1), difmapout, \
                                  pixsize, jmsuffix)
    write_difmappsrscript(source, 'combined', difmapout, \
                          pixsize, jmsuffix)
    if saveagain:
        difmapout.write("wobs " + rootdir + "/" + experiment + "/noise" + \
                        source + ".uvf\n")
    difmapout.write("exit\n")
    difmapout.close()
    os.system(difmappath + " < difmap_commands")

# WRITE_DIFMAPPSRSCRIPT
def write_difmappsrscript(source, bands, difmapout, pixsize, jmsuffix):
    difmapout.write("clrmod true\n")
    difmapout.write("unshift\n")
    difmapout.write("shift -peakx,-peaky\n")
    difmapout.write("mapsize 1024,0.1\n")
    difmapout.write("pkflux = peak(flux,max)\n")
    difmapout.write("addcmp pkflux, true, finepeakx, finepeaky, true, 0, " + \
                    "false, 1, false, 0, false, 0, 0, 0\n")
    difmapout.write("mapsize 1024," + str(pixsize) + "\n")
    difmapout.write("modelfit 50\n")
    difmapout.write("rmsflux = imstat(rms)\n")
    difmapout.write("restore\n")
    difmapout.write("pkflux = peak(flux,max)\n")
    difmapout.write("ilevs = pkflux/rmsflux\n")
    difmapout.write("lowlev = 300.0/ilevs\n")
    difmapout.write("loglevs lowlev\n")
    imagefile = rootdir + '/' + experiment + '/' + source + '.' + bands + \
                '.image.fits' + jmsuffix
    if os.path.exists(imagefile):
        os.system("rm -f " + imagefile)
    difmapout.write("wmap " + imagefile + "\n")
    difmapout.write("unshift\n")
    difmapout.write("wmod\n")
    difmapout.write("print rmsflux\n")
    difmapout.write("print imstat(bmin)\n")
    difmapout.write("print imstat(bmaj)\n")
    difmapout.write("print imstat(bpa)\n")
