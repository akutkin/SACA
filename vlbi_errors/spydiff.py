import os
import datetime
import numpy as np
import copy
from utils import degree_to_rad
from components import DeltaComponent, CGComponent, EGComponent


def clean_n(fname, outfname, stokes, mapsize_clean, niter=100,
            path_to_script=None, mapsize_restore=None, beam_restore=None,
            outpath=None, shift=None, show_difmap_output=False,
            txt_windows=None, clean_box=None):
    if outpath is None:
        outpath = os.getcwd()

    if not mapsize_restore:
        mapsize_restore = mapsize_clean

    difmapout = open("difmap_commands", "w")
    difmapout.write("observe " + fname + "\n")
    if txt_windows is not None:
        difmapout.write("rwin " + str(txt_windows) + "\n")

    if clean_box is not None:
        difmapout.write("addwin " + str(clean_box[0]) + ', ' +
                        str(clean_box[1]) + ', ' + str(clean_box[2]) + ', ' +
                        str(clean_box[3]) + "\n")

    difmapout.write("mapsize " + str(mapsize_clean[0] * 2) + ', ' +
                    str(mapsize_clean[1]) + "\n")
    difmapout.write("@" + path_to_script + " " + stokes + ", " + str(niter) + "\n")
    if beam_restore:
        difmapout.write("restore " + str(beam_restore[0]) + ', ' +
                        str(beam_restore[1]) + ', ' + str(beam_restore[2]) +
                        "\n")
    difmapout.write("mapsize " + str(mapsize_restore[0] * 2) + ', ' +
                    str(mapsize_restore[1]) + "\n")
    if outpath is None:
        outpath = os.getcwd()
    if shift is not None:
        difmapout.write("shift " + str(shift[0]) + ', ' + str(shift[1]) + "\n")
    difmapout.write("wmap " + os.path.join(outpath, outfname) + "\n")
    difmapout.write("exit\n")
    difmapout.close()
    # TODO: Use subprocess for silent cleaning?
    shell_command = "difmap < difmap_commands 2>&1"
    if not show_difmap_output:
        shell_command += " >/dev/null"
    os.system(shell_command)


# TODO: add ``shift`` argument, that shifts image before cleaning. It must be
# more accurate to do this in difmap. Or add such method in ``UVData`` that
# multiplies uv-data on exp(-1j * (u*x_shift + v*y_shift)).
def clean_difmap(fname, outfname, stokes, mapsize_clean, path=None,
                 path_to_script=None, mapsize_restore=None, beam_restore=None,
                 outpath=None, shift=None, show_difmap_output=False,
                 command_file=None, clean_box=None):
    """
    Map self-calibrated uv-data in difmap.
    :param fname:
        Filename of uv-data to clean.
    :param outfname:
        Filename with CCs.
    :param stokes:
        Stokes parameter 'i', 'q', 'u' or 'v'.
    :param mapsize_clean: (optional)
        Parameters of map for cleaning (map size, pixel size). If ``None``
        then use those of map in map directory (not bootstrapped).
        (default: ``None``)
    :param path: (optional)
        Path to uv-data to clean. If ``None`` then use current directory.
        (default: ``None``)
    :param path_to_script: (optional)
        Path to ``clean`` difmap script. If ``None`` then use current directory.
        (default: ``None``)
    :param mapsize_restore: (optional)
        Parameters of map for restoring CC (map size, pixel size). If
        ``None`` then use naitive. (default: ``None``)
    :param beam_restore: (optional)
        Beam parameter for restore map (bmaj, bmin, bpa). If ``None`` then use
        the same beam as in cleaning. (default: ``None``)
    :param outpath: (optional)
        Path to file with CCs. If ``None`` then use ``path``.
        (default: ``None``)
    :param shift: (optional)
        Iterable of 2 values - shifts in both directions - East & North [mas].
        If ``None`` then don't shift. (default: ``None``)
    :param show_difmap_output: (optional)
        Show difmap output? (default: ``False``)
    :param command_file: (optional)
        Script file name to store `difmap` commands. If ``None`` then use
        ``difmap_commands``. (default: ``None``)

    """
    if path is None:
        path = os.getcwd()
    if outpath is None:
        outpath = os.getcwd()

    if not mapsize_restore:
        mapsize_restore = mapsize_clean

    if command_file is None:
        command_file = "difmap_commands"

    difmapout = open(command_file, "w")
    difmapout.write("observe " + os.path.join(path, fname) + "\n")
    # if shift is not None:
    #     difmapout.write("shift " + str(shift[0]) + ', ' + str(shift[1]) + "\n")
    difmapout.write("mapsize " + str(mapsize_clean[0] * 2) + ', ' +
                    str(mapsize_clean[1]) + "\n")
    if clean_box is not None:
        difmapout.write("addwin " + str(clean_box[0]) + ', ' +
                        str(clean_box[1]) + ', ' + str(clean_box[2]) + ', ' +
                        str(clean_box[3]) + "\n")
    difmapout.write("@" + path_to_script + " " + stokes + "\n")
    if beam_restore:
        difmapout.write("restore " + str(beam_restore[0]) + ', ' +
                        str(beam_restore[1]) + ', ' + str(beam_restore[2]) +
                        "\n")
    difmapout.write("mapsize " + str(mapsize_restore[0] * 2) + ', ' +
                    str(mapsize_restore[1]) + "\n")
    if outpath is None:
        outpath = path
    elif not outpath.endswith("/"):
        outpath = outpath + "/"
    if shift is not None:
        difmapout.write("shift " + str(shift[0]) + ', ' + str(shift[1]) + "\n")
    difmapout.write("wmap " + os.path.join(outpath, outfname) + "\n")
    difmapout.write("exit\n")
    difmapout.close()
    # TODO: Use subprocess for silent cleaning?
    shell_command = "/home/osh/Soft/difmap_2.4l_p1/./difmap < " + command_file + " 2>&1"
    if not show_difmap_output:
        shell_command += " >/dev/null"
    os.system(shell_command)


def import_difmap_model(mdl_fname, mdl_dir=None):
    """
    Function that reads difmap-format model and returns list of ``Components``
    instances.

    :param mdl_fname:
        File name with difmap model.
    :param mdl_dir: (optional)
        Directory with difmap model. If ``None`` then use CWD. (default:
        ``None``)
    :return:
        List of ``Components`` instances.
    """
    if mdl_dir is None:
        mdl_dir = os.getcwd()
    mdl = os.path.join(mdl_dir, mdl_fname)
    mdlo = open(mdl)
    lines = mdlo.readlines()
    comps = list()
    for line in lines:
        if line.startswith('!'):
            continue
        line = line.strip('\n ')
        try:
            flux, radius, theta, major, axial, phi, type_, freq, spec = line.split()
        except ValueError:
            try:
                flux, radius, theta, major, axial, phi, type_ = line.split()
            except ValueError:
                try:
                    flux, radius, theta = line.split()
                except ValueError:
                    print "Problem parsing line :\n"
                    print line
                    raise ValueError
                axial = 1.0
                major = 0.0
                type_ = 0

        list_fixed = list()
        if flux[-1] != 'v':
            list_fixed.append('flux')

        x = -float(radius[:-1]) * np.sin(np.deg2rad(float(theta[:-1])))
        y = -float(radius[:-1]) * np.cos(np.deg2rad(float(theta[:-1])))
        flux = float(flux[:-1])

        if int(type_) == 0:
            comp = DeltaComponent(flux, x, y)
        elif int(type_) == 1:

            try:
                bmaj = float(major)
                list_fixed.append('bmaj')
            except ValueError:
                bmaj = float(major[:-1])
            if float(axial[:-1]) == 1:
                comp = CGComponent(flux, x, y, bmaj, fixed=list_fixed)
            else:
                try:
                    e = float(axial)
                except ValueError:
                    e = float(axial[:-1])
                try:
                    bpa = np.deg2rad(float(phi)) + np.pi / 2.
                except ValueError:
                    bpa = np.deg2rad(float(phi[:-1])) + np.pi / 2.
                comp = EGComponent(flux, x, y, bmaj, e, bpa, fixed=list_fixed)
        else:
            raise NotImplementedError("Only CC, CG & EG are implemented")
        comps.append(comp)
    return comps


def core_alpha_delta(mfile):
    fid = open(mfile)
    lines = fid.readlines()
    fid.close()
    _, r, t = np.array([l.replace('v','').split()[:3] for l in lines
                    if not l.startswith('!')], dtype=float)[0]

    return r*np.cos(t*np.pi/180.0 + np.pi/2), -r*np.sin(t*np.pi/180.0 + np.pi/2)


def export_difmap_model(comps, out_fname, freq_hz):
    """
    Function that export iterable of ``Component`` instances to difmap format
    model file.
    :param comps:
        Iterable of ``Component`` instances.
    :param out_fname:
        Path for saving file.
    """
    with open(out_fname, "w") as fo:
        fo.write("! Flux (Jy) Radius (mas)  Theta (deg)  Major (mas)  Axial ratio   Phi (deg) T\n\
! Freq (Hz)     SpecIndex\n")
        for comp in comps:
            if isinstance(comp, EGComponent):
                if comp.size == 4:
                    # Jy, mas, mas, mas
                    flux, x, y, bmaj = comp.p
                    e = "1.00000"
                    bpa = "000.000"
                    type = "1"
                    bmaj = "{}v".format(bmaj)
                elif comp.size == 6:
                    # Jy, mas, mas, mas, -, deg
                    flux, x, y, bmaj, e, bpa = comp.p
                    e = "{}v".format(e)
                    bpa = "{}v".format((bpa-np.pi/2)/degree_to_rad)
                    bmaj = "{}v".format(bmaj)
                    type = "1"
                else:
                    raise Exception
            elif isinstance(comp, DeltaComponent):
                flux, x, y = comp.p
                e = "1.00000"
                bmaj = "0.0000"
                bpa = "000.000"
                type = "0"
            else:
                raise Exception
            # mas
            r = np.hypot(x, y)
            # rad
            theta = np.arctan2(-x, -y)
            theta /= degree_to_rad
            fo.write("{}v {}v {}v {} {} {} {} {} 0\n".format(flux, r, theta,
                                                           bmaj, e, bpa, type,
                                                           freq_hz))


def append_component_to_difmap_model(comp, out_fname, freq_hz):
    """
    :param comp:
        Instance of ``Component``.
    :param fname:
        File with difmap model.
    """
    with open(out_fname, "a") as fo:
        if isinstance(comp, EGComponent):
            if comp.size == 4:
                # Jy, mas, mas, mas
                flux, x, y, bmaj = comp.p
                e = "1.00000"
                bpa = "000.000"
                type = "1"
                bmaj = "{}v".format(bmaj)
            elif comp.size == 6:
                # Jy, mas, mas, mas, -, deg
                flux, x, y, bmaj, e, bpa = comp.p
                e = "{}v".format(e)
                bpa = "{}v".format((bpa - np.pi / 2) / degree_to_rad)
                bmaj = "{}v".format(bmaj)
                type = "1"
            else:
                raise Exception
        elif isinstance(comp, DeltaComponent):
            flux, x, y = comp.p
            e = "1.00000"
            bmaj = "0.0000"
            bpa = "000.000"
            type = "0"
        else:
            raise Exception
        # mas
        r = np.hypot(x, y)
        # rad
        theta = np.arctan2(-x, -y)
        theta /= degree_to_rad
        fo.write("{}v {}v {}v {} {} {} {} {} 0\n".format(flux, r, theta,
                                                         bmaj, e, bpa, type,
                                                         freq_hz))


def difmap_model_flux(mdl_file):
    """
    Returns flux of the difmap model.

    :param mdl_file:
        Path to difmap model file.
    :return:
        Sum of all component fluxes [Jy].
    """
    comps = import_difmap_model(mdl_file)
    fluxes = [comp.p[0] for comp in comps]
    return sum(fluxes)


def find_ids(comps0, comps1):
    assert len(comps0) == len(comps1)
    ids = list()
    for comp in comps0:
        ids.append(comps1.index(comp))
    return ids


def sort_components_by_distance_from_cj(mdl_path, freq_hz, n_check_for_core=2,
                                        outpath=None, perc_distant=75,
                                        only_indicate=False):
    """
    Function that re-arrange components in a such a way that closest to "counter
    jet" components goes first. If components were already sorted by distance
    from "counter-jet" then does nothing and return ``False``.

    :param mdl_path:
        Path to difmap-format model file.
    :param freq_hz:
        Frequency in Hz.
    :param n_check_for_core: (optional)
        Number of closest to phase center components to check for being on "cj"
        side. (default: ``2``)
    :param outpath: (optional)
        Path to save re-arranged model. If ``None`` then re-write ``mdl_path``.
        (default: ``None``)
    :param perc_distant: (optional)
        Percentile of the component's distance distribution to use as border
        line that defines "distant" component. Position of such components is
        then used to compare polar angles of "cj-candidates". (default: ``75``)
    :param only_indicate: (optional)
        Boolean - ust indicate by the returned value or also do re-arrangement
        of components? (default: ``False``)
    :return:
        Boolean if there any "cj"-components that weren't sorted by distance
        from "CJ".
    """
    mdl_dir, mdl_fname = os.path.split(mdl_path)
    comps = import_difmap_model(mdl_fname, mdl_dir)
    comps = sorted(comps, key=lambda x: np.hypot(x.p[1], x.p[2]))
    comps_c = import_difmap_model(mdl_fname, mdl_dir)
    r = [np.hypot(comp.p[1], comp.p[2]) for comp in comps]
    dist = np.percentile(r, perc_distant)
    # Components that are more distant from phase center than ``perc_distant``
    # of components
    remote_comps = [comp for r_, comp in zip(r, comps) if r_ > dist]
    theta_remote = np.mean([np.arctan2(-comp.p[1], -comp.p[2])/degree_to_rad for
                            comp in remote_comps])

    found_cj_comps = list()
    for i in range(1, n_check_for_core+1):
        comp = comps[i]
        # This checks that cj component have different PA
        if abs(np.arctan2(-comp.p[1], -comp.p[2])/degree_to_rad -
               theta_remote) > 90.:
            print("Found cj components - {}".format(comp))
            found_cj_comps.append(comp)

    result_comps = list()
    if found_cj_comps:
        for comp in found_cj_comps:
            comps.remove(comp)

        for comp in sorted(found_cj_comps,
                           key=lambda x: np.hypot(x.p[1], x.p[2])):
            result_comps.append(comp)

        # Now check if cj-components were not first ones
        is_changed = False
        sorted_found_cj_comps = sorted(found_cj_comps,
                                       key=lambda x: np.hypot(x.p[1], x.p[2]))
        for comp1, comp2 in zip(comps_c[:len(sorted_found_cj_comps)],
                                sorted_found_cj_comps[::-1]):
            if comp1 != comp2:
                is_changed = True
        # If all job was already done just return
        if not is_changed:
            print("Components are already sorted")
            return False

    for comp in comps:
        result_comps.append(comp)

    if outpath is None:
        outpath = mdl_path
    if not only_indicate:
        export_difmap_model(result_comps, outpath, freq_hz)
    return bool(found_cj_comps)


def sum_components(comp0, comp1, type="eg"):
    print("Summing components {} and {}".format(comp0, comp1))
    flux = comp0.p[0] + comp1.p[0]
    x = (comp0.p[0] * comp0.p[1] + comp1.p[0] * comp1.p[1]) / flux
    y = (comp0.p[0] * comp0.p[2] + comp1.p[0] * comp1.p[2]) / flux
    bmaj = (comp0.p[0] * comp0.p[3] + comp1.p[0] * comp1.p[3]) / flux
    if type == "cg":
        component = CGComponent(flux, x, y, bmaj)
    elif type == "eg":
        dx = comp0.p[1]-comp1.p[1]
        dy = comp0.p[2]-comp1.p[2]
        bpa = 180*np.arctan(dx/dy)/np.pi
        e = 0.5*(comp0.p[3]+comp1.p[3])/np.hypot(comp0.p[1]-comp1.p[1], comp0.p[2]-comp1.p[2])
        component = EGComponent(flux, x, y, bmaj, e, bpa)
    else:
        raise Exception
    print("Sum = {}".format(component))
    return component


def component_joiner_serial(difmap_model_file, beam_size, freq_hz,
                            distance_to_join_max=1.0, outname=None, new_type="eg"):
    joined = False
    comps = import_difmap_model(difmap_model_file)
    print("Len = {}".format(len(comps)))
    new_comps = list()
    skip_next = False
    for i, comp0 in enumerate(comps):
        print("{} and {}".format(i, i+1))
        if skip_next:
            skip_next = False
            print ("Skipping {}th component".format(i))
            continue
        try:
            comp1 = comps[i+1]
        except IndexError:
            new_comps.append(comp0)
            print("Writing component {} and exiting".format(i))
            break
        print("Fluxes: {} and {}".format(comp0.p[0], comp1.p[0]))

        if i != 0:
            distance_before = max(np.hypot(comp0.p[1]-comps[i-1].p[1], comp0.p[2]-comps[i-1].p[2]),
                                  np.hypot(comp1.p[1]-comps[i-1].p[1], comp1.p[2]-comps[i-1].p[2]))
        else:
            distance_before = 10**10
        print("Distance before = {}".format(distance_before))

        distance_between = np.hypot(comp0.p[1]-comp1.p[1],
                                    comp0.p[2]-comp1.p[2])
        print("Distance between = {}".format(distance_between))

        if i < len(comps)-2:
            distance_after = max(np.hypot(comp0.p[1] - comps[i + 2].p[1],
                                          comp0.p[2] - comps[i + 2].p[2]),
                                 np.hypot(comp1.p[1] - comps[i + 2].p[1],
                                          comp1.p[2] - comps[i + 2].p[2]))
        else:
            distance_after = 10000.0
        print("Distance after = {}".format(distance_after))

        if distance_before > beam_size and\
                        distance_after > beam_size and\
                        distance_between < distance_to_join_max*beam_size and\
                len(comp0) == len(comp1):
            joined = True
            new_comps.append(sum_components(comp0, comp1, type=new_type))
            skip_next = True
        else:
            new_comps.append(comp0)
            skip_next = False

    if outname is None:
            outname = difmap_model_file

    export_difmap_model(new_comps, outname, freq_hz)
    return joined





# FIXME: Check if it works to ``DeltaComponent``!
def transform_component(difmap_model_file, new_type, freq_hz, comp_id=0,
                        outname=None, **kwargs):
    """
    Change component type in difmap model file.

    :param difmap_model_file:
        Path to difmap file with model.
    :param new_type:
        String that defines what is the new component type (delta, cg, eg).
    :param freq_hz:
        Frequency in Hz.
    :param comp_id: (optional)
        Number (ID) of component to transform. ``0`` means transform first
        component (core). (default: ``0``)
    :param outname: (optional)
        Where to save the result. If ``None`` then rewrite
        ``difmap_model_file``.
    :param kwargs: (optional)
        Arguments needed for transforming simple component to more complex.
    """
    if new_type not in ("delta", "cg", "eg"):
        raise Exception
    comps = import_difmap_model(difmap_model_file)
    comp_to_transform = comps[comp_id]
    new_comps = list()
    for comp in comps[:comp_id]:
        new_comps.append(comp)
    if new_type == "delta":
        new_comps.append(comp_to_transform.to_delta())
    elif new_type == "cg":
        new_comps.append(comp_to_transform.to_cirular(**kwargs))
    else:
        new_comps.append(comp_to_transform.to_elliptic(**kwargs))
    for comp in comps[comp_id+1:]:
        new_comps.append(comp)

    if outname is None:
        outname = difmap_model_file
    export_difmap_model(new_comps, outname, freq_hz)


def modelfit_difmap(fname, mdl_fname, out_fname, niter=50, stokes='i',
                    path=None, mdl_path=None, out_path=None,
                    show_difmap_output=False):
    """
    Modelfit self-calibrated uv-data in difmap.

    :param fname:
        Filename of uv-data to modelfit.
    :param mdl_fname:
        Filename with model.
    :param out_fname:
        Filename with output file with model.
    :param stokes: (optional)
        Stokes parameter 'i', 'q', 'u' or 'v'. (default: ``i``)
    :param path: (optional)
        Path to uv-data to modelfit. If ``None`` then use current directory.
        (default: ``None``)
    :param mdl_path: (optional)
        Path file with model. If ``None`` then use current directory.
        (default: ``None``)
    :param out_path: (optional)
        Path to file with CCs. If ``None`` then use ``path``.
        (default: ``None``)
    """
    if path is None:
        path = os.getcwd()
    if mdl_path is None:
        mdl_path = os.getcwd()
    if out_path is None:
        out_path = os.getcwd()

    stamp = datetime.datetime.now()
    command_file = os.path.join(out_path, "difmap_commands_{}".format(stamp.isoformat()))
    # difmapout = open("difmap_commands", "w")
    difmapout = open(command_file, "w")
    difmapout.write("observe " + os.path.join(path, fname) + "\n")
    difmapout.write("select " + stokes + "\n")
    difmapout.write("rmodel " + os.path.join(mdl_path, mdl_fname) + "\n")
    difmapout.write("modelfit " + str(niter) + "\n")
    difmapout.write("wmodel " + os.path.join(out_path, out_fname) + "\n")
    difmapout.write("exit\n")
    difmapout.close()

    # TODO: Use subprocess?
    shell_command = "/home/osh/Soft/difmap_2.4l_p1/./difmap < " + command_file + " 2>&1"
    if not show_difmap_output:
        shell_command += " >/dev/null"
    os.system(shell_command)


def make_map_with_core_at_zero(mdl_file, uv_fits_fname, mapsize_clean,
                               path_to_script,
                               beam_restore=None,
                               outfname="shifted_cc.fits",
                               stokes="I"):
    """
    Function that shifts map in a way that core in new map will have zero
    coordinates.

    :param mdl_file:
        Path to difmap model file with the core being the first component.
    :param uv_fits_fname:
        Path to UV FITS data.
    :param mapsize_clean:
        Iterable of number of pixels and pixel size [mas].
    :param path_to_script:
        Path to D.Homan difmap final CLEAN script.
    :param outfname:
        Where to save new FITS file with map.
    :param stokes: (optional)
        Stokes parameter. (default: ``I``)
    """
    mdl_dir, mdl_fn = os.path.split(mdl_file)
    uv_dir, uv_fn = os.path.split(uv_fits_fname)

    ra_mas, dec_mas = core_alpha_delta(mdl_file)
    shift = (ra_mas, dec_mas)
    clean_difmap(uv_fn, outfname, stokes, mapsize_clean, uv_dir, path_to_script,
                 beam_restore=beam_restore, shift=shift)


# # DIFMAP_MAPPSR
# def difmap_mappsr(source, isll, centre_ra_deg, centre_dec_deg, uvweightstr,
#                   experiment, difmappath, uvprefix, uvsuffix, jmsuffix, \
#                   saveagain):
#     difmapout = open("difmap_commands", "w")
#     difmapout.write("float pkflux\n")
#     difmapout.write("float peakx\n")
#     difmapout.write("float peaky\n")
#     difmapout.write("float finepeakx\n")
#     difmapout.write("float finepeaky\n")
#     difmapout.write("float rmsflux\n")
#     difmapout.write("integer ilevs\n")
#     difmapout.write("float lowlev\n")
#     difmapout.write("obs " + sourceuvfile + "\n")
#     difmapout.write("mapsize 1024," + str(pixsize) + "\n")
#     difmapout.write("select ll,1,2,3,4\n")
#     difmapout.write("invert\n");
#     difmapout.write("wtscale " + str(threshold) + "\n")
#     difmapout.write("obs " + sourceuvfile + "\n")
#     if reweight:
#         difmapout.write("uvaver 180,true\n")
#     else:
#         difmapout.write("uvaver 20\n")
#     difmapout.write("mapcolor none\n")
#     difmapout.write("uvweight " + uvweightstr + "\n")
#
#     #Find the peak from the combined first
#     if isll:
#         difmapout.write("select ll,1,2,3,4\n")
#     else:
#         difmapout.write("select i,1,2\n")
#     if experiment == 'v190k':
#         difmapout.write("select rr,1,2,3,4\n")
#     difmapout.write("peakx = peak(x,max)\n")
#     difmapout.write("peaky = peak(y,max)\n")
#     difmapout.write("shift -peakx,-peaky\n")
#     difmapout.write("mapsize 1024,0.1\n")
#     difmapout.write("finepeakx = peak(x,max)\n")
#     difmapout.write("finepeaky = peak(y,max)\n")
#
#     #Do each individual band, one at a time
#     maxif = 4
#     pols = ['rr','ll']
#     for i in range(maxif):
#         for p in pols:
#             difmapout.write("select " + p + "," + str(i+1) + "\n")
#             write_difmappsrscript(source, p + "." + str(i+1), difmapout, \
#                                   pixsize, jmsuffix)
#     write_difmappsrscript(source, 'combined', difmapout, \
#                           pixsize, jmsuffix)
#     if saveagain:
#         difmapout.write("wobs " + rootdir + "/" + experiment + "/noise" + \
#                         source + ".uvf\n")
#     difmapout.write("exit\n")
#     difmapout.close()
#     os.system(difmappath + " < difmap_commands")
#
# # WRITE_DIFMAPPSRSCRIPT
# def write_difmappsrscript(source, bands, difmapout, pixsize, jmsuffix):
#     difmapout.write("clrmod true\n")
#     difmapout.write("unshift\n")
#     difmapout.write("shift -peakx,-peaky\n")
#     difmapout.write("mapsize 1024,0.1\n")
#     difmapout.write("pkflux = peak(flux,max)\n")
#     difmapout.write("addcmp pkflux, true, finepeakx, finepeaky, true, 0, " + \
#                     "false, 1, false, 0, false, 0, 0, 0\n")
#     difmapout.write("mapsize 1024," + str(pixsize) + "\n")
#     difmapout.write("modelfit 50\n")
#     difmapout.write("rmsflux = imstat(rms)\n")
#     difmapout.write("restore\n")
#     difmapout.write("pkflux = peak(flux,max)\n")
#     difmapout.write("ilevs = pkflux/rmsflux\n")
#     difmapout.write("lowlev = 300.0/ilevs\n")
#     difmapout.write("loglevs lowlev\n")
#     imagefile = rootdir + '/' + experiment + '/' + source + '.' + bands + \
#                 '.image.fits' + jmsuffix
#     if os.path.exists(imagefile):
#         os.system("rm -f " + imagefile)
#     difmapout.write("wmap " + imagefile + "\n")
#     difmapout.write("unshift\n")
#     difmapout.write("wmod\n")
#     difmapout.write("print rmsflux\n")
#    difmapout.write("print imstat(bmin)\n")
#    difmapout.write("print imstat(bmaj)\n")
#    difmapout.write("print imstat(bpa)\n")

if __name__ == "__main__":
    print("Spydiff module")