"""
@author: Anthony Soulain (University of Sydney)

-------------------------------------------------------------------------
AMICAL: Aperture Masking Interferometry Calibration and Analysis Library
-------------------------------------------------------------------------

Instruments and mask informations.
--------------------------------------------------------------------
"""
import sys

import numpy as np
from termcolor import cprint

from amical.tools import mas2rad

if sys.version_info >= (3, 9):
    import importlib.resources as importlib_resources
else:
    import importlib_resources


def get_mask(ins, mask, first=0):
    """Return dictionnary containning saved informations about masks."""

    pupil_visir = 8.0
    pupil_visir_mm = 17.67
    off = 0.3
    dic_mask = {
        "NIRISS": {
            "g7": np.array(
                [
                    [0, -2.64],
                    [-2.28631, 0],
                    [2.28631, -1.32],
                    [-2.28631, 1.32],
                    [-1.14315, 1.98],
                    [2.28631, 1.32],
                    [1.14315, 1.98],
                ]
            ),
            "g7_bis": np.array(
                [
                    [0, 2.9920001],
                    [2.2672534, 0.37400016],
                    [-2.2672534, 1.6829998],
                    [2.2672534, -0.93499988],
                    [1.1336316, -1.5895000],
                    [-2.2672534, -0.93500012],
                    [-1.1336313, -1.5895000],
                ]
            ),
            "g7_sb": np.array(
                [
                    [0, -2.64],  # 0
                    [-2.28631, 0],  # 1
                    [-2.28631 + off, 0],
                    [-2.28631 - off / np.sqrt(2), 0 + off / np.sqrt(2)],
                    [-2.28631 - off / np.sqrt(2), 0 - off / np.sqrt(2)],
                    [2.28631, -1.32],  # 2
                    [-2.28631, 1.32],  # 3
                    [-1.14315, 1.98],  # 4
                    [-1.14315 + off, 1.98],
                    [-1.14315 - off / np.sqrt(2), 1.98 + off / np.sqrt(2)],
                    [-1.14315 - off / np.sqrt(2), 1.98 - off / np.sqrt(2)],
                    [2.28631, 1.32],  # 5
                    [2.28631 + off, 1.32],
                    [2.28631 - off / np.sqrt(2), 1.32 + off / np.sqrt(2)],
                    [2.28631 - off / np.sqrt(2), 1.32 - off / np.sqrt(2)],
                    [1.14315, 1.98],  # 6
                ]
            ),
        },
        "GLINT": {
            "g4": np.array(
                [[2.725, 2.317], [-2.812, 1.685], [-2.469, -1.496], [-0.502, -2.363]]
            )
        },
        "NACO": {
            "g7": np.array(
                [
                    [-3.51064, -1.99373],
                    [-3.51064, 2.49014],
                    [-1.56907, 1.36918],
                    [-1.56907, 3.61111],
                    [0.372507, -4.23566],
                    [2.31408, 3.61111],
                    [4.25565, 0.248215],
                ]
            )
            * (8 / 10.0),
        },
        "SPHERE": {
            "g7": 1.05
            * np.array(
                [
                    [-1.46, 2.87],
                    [1.46, 2.87],
                    [-2.92, 0.34],
                    [-1.46, -0.51],
                    [-2.92, -1.35],
                    [2.92, -1.35],
                    [0, -3.04],
                ]
            )
        },
        "SPHERE-IFS": {
            "g7": 1
            * np.array(
                [
                    [-2.07, 2.71],
                    [0.98, 3.27],
                    [-3.11, -0.2],
                    [-1.43, -0.81],
                    [-2.79, -1.96],
                    [3.3, -0.85],
                    [0.58, -3.17],
                ]
            )
        },
        "VISIR": {
            "g7": (pupil_visir / pupil_visir_mm)
            * np.array(
                [
                    [-5.707, -2.885],
                    [-5.834, 3.804],
                    [0.099, 7.271],
                    [7.989, 0.422],
                    [3.989, -6.481],
                    [-3.790, -6.481],
                    [-1.928, -2.974],
                ]
            ),
        },
        "VAMPIRES": {
            "g18": np.array(
                [
                    [0.821457, 2.34684],
                    [-2.34960, 1.49034],
                    [-2.54456, 2.55259],
                    [1.64392, 3.04681],
                    [2.73751, -0.321102],
                    [1.38503, -3.31443],
                    [-3.19337, -1.68413],
                    [3.05126, 0.560011],
                    [-2.76083, 1.14035],
                    [3.02995, -1.91449],
                    [0.117786, 3.59025],
                    [-0.802156, 3.42140],
                    [-1.47228, -3.28982],
                    [-1.95968, -0.634178],
                    # [-3.29085, -1.15300],
                    [0.876319, -3.13328],
                    [2.01253, -1.55220],
                    [-2.07847, -2.57755],
                ]
            )
        },
        "ERIS": {
            "g7": np.array(
                [
                    [-4.5396,0.0000],
                    [-2.0176,-4.3680],
                    [-2.0176,4.3680],
                    [-0.7566,2.1840],
                    [0.5044,4.3680],
                    [4.2874,-2.1840],
                    [4.2874,2.1840]
                ]
            )*(4/6),
            "K7": np.array(
                [
                    [-1.43208,-2.90128],
                    [2.84935,-1.54168],
                    [-3.03715,0.08430],
                    [-0.46429,1.47584],
                    [-1.27767,2.97370],
                    [0.41675,2.93229],
                    [2.92930,1.40173],
                ]
            ),
            "K9": np.array(
                [
                    [-1.29141,-2.50426],
                    [-0.05651,-2.55380],
                    [-3.07906,-1.35257],
                    [2.50008,-0.53502],
                    [3.16390,0.49262],
                    [2.03608,2.69131],
                    [-1.08498,1.75589],
                    [-2.97730,0.77755],
                    [-1.65540,2.85226],
                ]
            ),
            "K23": np.array(
                [
                    [-0.84358,-3.30595],
                    [0.79280,-3.31551],
                    [1.34313,-3.31995],
                    [-2.20666,-2.82978],
                    [-0.83618,-2.35345],
                    [2.16211,-2.85223],
                    [2.71214,-1.91010],
                    [-3.00628,-1.39644],
                    [0.25673,-1.41449],
                    [2.99041,-1.43703],
                    [-3.54770,-0.44685],
                    [-1.35529,0.47650],
                    [-3.26345,0.02218],
                    [-2.98803,1.43356],
                    [3.55430,0.45062],
                    [3.28338,0.92344],
                    [2.73827,1.87316],
                    [-1.34629,2.36821],
                    [-2.16469,2.84510],
                    [1.65331,2.82642],
                    [-0.25246,3.31036],
                    [0.28739,3.30683],
                    [1.38649,3.30306],
                ]
            ),
            "L7": np.array(
                [
                    [-1.44072,-2.93037],
                    [2.87469,-1.55808],
                    [-3.07028,0.07822],
                    [-0.46611,1.48670],
                    [-1.28940,3.00279],
                    [0.42100,2.95654],
                    [2.95499,1.40318],
                ]
            ),
            "L9": np.array(
                [
                    [-1.29555,-2.53343],
                    [-0.05646,-2.58319],
                    [-3.11298,-1.37206],
                    [2.52870,-0.54913],
                    [3.20219,0.50577],
                    [2.05802,2.71554],
                    [-1.09319,1.77315],
                    [-3.01157,0.78772],
                    [-1.67153,2.88476],
                ]
            ),
            "L23": np.array(
                [
                    [-0.84885,-3.33614],
                    [0.80448,-3.34898],
                    [1.35809,-3.35335],
                    [-2.22389,-2.85410],
                    [-0.84147,-2.37481],
                    [2.18000,-2.87942],
                    [2.73689,-1.92905],
                    [-3.03943,-1.41311],
                    [0.26520,-1.42765],
                    [3.01884,-1.44623],
                    [-3.58787,-0.45518],
                    [-1.37493,0.48392],
                    [-3.30736,0.02382],
                    [-3.02482,1.44928],
                    [3.58964,0.45796],
                    [3.31141,0.93319],
                    [2.76297,1.89113],
                    [-1.35485,2.39182],
                    [-2.18354,2.87851],
                    [1.67155,2.85251],
                    [-0.25111,3.33861],
                    [0.29187,3.33517],
                    [1.40134,3.33522],
                ]
            ),
            # "g7": np.array(
            #     [
            #         [-2.3238596029232887, 5.400799599141509],
            #         [5.4701252465863766, 2.6227781594455295],
            #         [-5.470444847510135, 0.008045431047894248],
            #         [-0.8236951337569205, -2.718941066171585],
            #         [-2.4172867296646663, -5.423960614228495],
            #         [0.7045175281691775, -5.456649639917031],
            #         [5.428086790622207, -2.800663777493688]
            #     ]
            # )*(2/3),
        # "g7": np.array(
        #                 [
        #                     [0.93691,2.75225],
        #                     [-0.31424,2.75873],
        #                     [2.78623,1.65020],
        #                     [-2.82952,0.62594],
        #                     [-3.46482,-0.43722],
        #                     [-2.24608,-2.62188],
        #                     [0.87856,-1.56520],
        #                     [2.75854,-0.50853],
        #                     [1.49442,-2.65429]
        #                 ]
        #             ),
            "g9": np.array(
                [
                    [-4.5116,-2.1350],
                    [-4.4565,1.0265],
                    [-2.5756,4.1562],
                    [-1.8007,-3.7635],
                    [-1.6903,2.5595],
                    [0.0251,-3.7954],
                    [2.9016,4.0606],
                    [3.7317,-0.6976],
                    [4.6721,0.8672]
                ]
            ),
            "g23": np.array(
                [
                    [-5.2621,-0.8134],
                    [-4.8709,-0.1020],
                    [-4.5092,2.0150],
                    [-4.4209,-2.2021],
                    [-3.3356,4.1491],
                    [-3.1590,-4.2851],
                    [-2.1032,3.4717],
                    [-2.0443,0.6603],
                    [-1.1441,-3.5397],
                    [-1.1147,-4.9454],
                    [-0.5090,4.9114],
                    [0.3029,4.9284],
                    [0.4501,-2.1000],
                    [1.3208,-4.8944],
                    [1.9265,4.9624],
                    [2.1326,-4.8774],
                    [2.3471,4.2681],
                    [3.3356,-4.1491],
                    [4.0002,2.8964],
                    [4.1180,-2.7264],
                    [4.5092,-2.0150],
                    [4.8415,1.5077],
                    [5.2621,0.8134]
                ]
            ),
        }
    }

    #

    try:
        xycoords = dic_mask[ins][mask]
        nrand = [first]
        for x in np.arange(len(xycoords)):
            if x not in nrand:
                nrand.append(x)
        xycoords_sel = xycoords[nrand]
    except KeyError:
        cprint(f"\n-- Error: maskname ({mask}) unknown for {ins}.", "red")
        xycoords_sel = None
    return xycoords_sel


def get_wavelength(ins, filtname):
    """Return dictionnary containning saved informations about filters."""
    from astropy.io import fits

    datadir = importlib_resources.files("amical") / "internal_data"
    YJfile = datadir / "ifs_wave_YJ.fits"
    YJHfile = datadir / "ifs_wave_YJH.fits"

    with fits.open(YJfile) as fd:
        wave_YJ = fd[0].data
    with fits.open(YJHfile) as fd:
        wave_YJH = fd[0].data

    dic_filt = {
        "NIRISS": {
            "F277W": [2.776, 0.715],
            "F380M": [3.828, 0.205],
            "F430M": [4.286, 0.202],
            "F480M": [4.817, 0.298],
        },
        'ERIS':{
            'J':[1.28,0.20],
            'H':[1.66,0.31],
            'Ks':[2.18,0.39],
            'Short-Lp':[3.32,0.43],
            'Lp':[3.79,0.60],
            'L-Broad':[3.57,1.04],
            'Mp':[4.78,0.58],
            'Pa-b':[1.282,0.021],
            'Fe-II':[1.644,0.020],
            'H2-cont':[2.068,0.064],
            'H2-1-0S':[2.120,0.020],
            'Br-g':[2.172,0.020],
            'K-peak':[2.198,0.099],
            'IB-2.42':[2.420,0.049],
            'IB-2.48':[2.479,0.051],
            'Br-a-cont':[3.965,0.108],
            'Br-a':[4.051,0.023]
        },
        "SPHERE": {
            "H2": [1.593, 0.052],
            "H3": [1.667, 0.054],
            "H4": [1.733, 0.057],
            "BB_H": [1.625, 0.290],
            "K1": [2.110, 0.102],
            "K2": [2.251, 0.109],
            "NB_BrG": [2.170, 0.031],
            "CntH": [1.573, 0.023],
            "CntK1": [2.091, 0.034],
            "CntK2": [2.266, 0.032],
        },
        "SPHERE-IFS": {"YJ": wave_YJ, "YH": wave_YJH},
        "GLINT": {"F155": [1.55, 0.01], "F430": [4.3, 0.01]},
        "VISIR": {"10_5_SAM": [10.56, 0.37], "11_3_SAM": [11.23, 0.55]},
        "VAMPIRES": {"750-50": [0.75, 0.05]},
    }

    if ins not in dic_filt.keys():
        raise KeyError(
            f"--- Error: instrument <{ins}> not found ---\n"
            "Available: %s" % list(dic_filt.keys())
        )
    if filtname not in dic_filt[ins]:
        raise KeyError(
            f"Missing input: filtname <{filtname}> not found for {ins} (Available: {list(dic_filt[ins])})"
        )
    return np.array(dic_filt[ins][filtname]) * 1e-6


def get_pixel_size(ins):
    saved_pixel_detector = {
        "NIRISS": 65.6,
        "ERIS": 13.,
        "SPHERE": 12.27,
        "VISIR": 45,
        "SPHERE-IFS": 7.46,
        "VAMPIRES": 6.475,
    }
    try:
        p = mas2rad(saved_pixel_detector[ins])
    except KeyError:
        p = np.NaN
    return p


def get_ifu_table(
    i_wl, filtname="YH", instrument="SPHERE-IFS", verbose=False, display=False
):
    """Get spectral information for the given instrumental IFU setup.
    `i_wl` can be an integer, a list of 2 integers (to get a range between those
    two) or a list of integers (>= 3) used to display the
    requested spectral channels."""
    wl = get_wavelength(instrument, filtname) * 1e6

    if verbose:
        print(f"\nInstrument: {instrument}, spectral range: {filtname}")
        print("-----------------------------")
        print(
            f"spectral coverage: {wl[0]:2.2f} - {wl[-1]:2.2f} µm (step = {np.diff(wl)[0]:2.2f})"
        )

    one_wl = True
    multiple_wl = False
    if isinstance(i_wl, list) & (len(i_wl) == 2):
        one_wl = False
        wl_range = wl[i_wl[0] : i_wl[1]]
        sp_range = np.arange(i_wl[0], i_wl[1], 1)
    elif isinstance(i_wl, list) & (len(i_wl) > 2):
        multiple_wl = True
        one_wl = False
    elif i_wl is None:
        one_wl = False
        sp_range = np.arange(len(wl))
        wl_range = wl

    if display:
        from matplotlib import pyplot as plt

        plt.figure(figsize=(4, 3))
        plt.title("--- SPECTRAL INFORMATION (IFU)---")
        plt.plot(wl, label="All spectral channels")
        if one_wl:
            plt.plot(
                np.arange(len(wl))[i_wl],
                wl[i_wl],
                "ro",
                label="Selected (%2.2f µm)" % wl[i_wl[0]],
            )
        elif multiple_wl:
            plt.plot(
                i_wl,
                wl[i_wl],
                "ro",
                label="Selected",
            )
        else:
            plt.plot(
                sp_range,
                wl_range,
                lw=5,
                alpha=0.5,
                label=f"Selected ({wl_range[0]:2.2f}-{wl_range[-1]:2.2f} µm)",
            )
        plt.legend()
        plt.xlabel("Spectral channel")
        plt.ylabel("Wavelength [µm]")
        plt.tight_layout()

    if one_wl:
        output = wl[i_wl]
    elif multiple_wl:
        output = np.array(wl[i_wl])
    else:
        output = wl_range
    return output
