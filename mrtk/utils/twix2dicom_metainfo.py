import mapvbvd
import numpy as np
'''Interpretation of the Siemens GSL functions.
Author: William Clarke <william.clarke@ndcn.ox.ac.uk>
Copyright (C) 2020 University of Oxford '''

import numpy as np


def class_ori(sag_comp, cor_comp, tra_comp, debug):
    ''' Python implementation of IDEA-VB17/n4/pkg/MrServers/MrMeasSrv/SeqFW/libGSL/fGSLClassOri.cpp
    Function to determine whether a normal vector describes a sagittal, coronal or transverse slice.
    Result:
        CASE = 0: Sagittal
        CASE = 1: Coronal
        CASE = 2: Transverse

    :param  sag_comp:   Sagittal component of normal vector
    :param  cor_comp:   Coronal component of normal vector
    :param  tra_comp:   Transverse component of normal vector

    :return:    case (0=Sagittal, 1=Coronal or 2=Transverse)
    '''

    if debug:
        print(f'Normal vector = {sag_comp : 10.7f} {cor_comp : 10.7f} {tra_comp : 10.7f}.')

    # Compute some temporary values
    abs_sag_comp     = np.abs(sag_comp)
    abs_cor_comp     = np.abs(cor_comp)
    abs_tra_comp     = np.abs(tra_comp)

    eq_sag_cor = np.isclose(abs_sag_comp, abs_cor_comp)
    eq_sag_tra = np.isclose(abs_sag_comp, abs_tra_comp)
    eq_cor_tra = np.isclose(abs_cor_comp, abs_tra_comp)

    # Determine the slice orientation (sag, cor, tra)
    if ((eq_sag_cor              & eq_sag_tra)             |
            (eq_sag_cor              & (abs_sag_comp < abs_tra_comp)) |
            (eq_sag_tra              & (abs_sag_comp > abs_cor_comp)) |
            (eq_cor_tra              & (abs_cor_comp > abs_sag_comp)) |
            ((abs_sag_comp > abs_cor_comp)  & (abs_sag_comp < abs_tra_comp)) |
            ((abs_sag_comp < abs_cor_comp)  & (abs_cor_comp < abs_tra_comp)) |
            ((abs_sag_comp < abs_tra_comp)  & (abs_tra_comp > abs_cor_comp)) |
            ((abs_cor_comp < abs_tra_comp)  & (abs_tra_comp > abs_sag_comp))):

        if debug:
            print('Mainly transverse.')
        case = 2  # Transverse

    elif ((eq_sag_cor              & (abs_sag_comp > abs_tra_comp)) |
            (eq_sag_tra              & (abs_sag_comp < abs_cor_comp)) |
            ((abs_sag_comp < abs_cor_comp)  & (abs_cor_comp > abs_tra_comp)) |
            ((abs_sag_comp > abs_tra_comp)  & (abs_sag_comp < abs_cor_comp)) |
            ((abs_sag_comp < abs_tra_comp)  & (abs_tra_comp < abs_cor_comp))):

        if debug:
            print('Mainly coronal.')
        case = 1  # Coronal

    elif ((eq_cor_tra              & (abs_cor_comp < abs_sag_comp)) |
            ((abs_sag_comp > abs_cor_comp)  & (abs_sag_comp > abs_tra_comp)) |
            ((abs_cor_comp > abs_tra_comp)  & (abs_cor_comp < abs_sag_comp)) |
            ((abs_cor_comp < abs_tra_comp)  & (abs_tra_comp < abs_sag_comp))):

        if debug:
            print('Mainly sagittal.')
        case = 0  # Sagittal

    else:    # Invalid slice orientation...
        raise ValueError('Error: Invalid slice orientation')

    return case


def calc_prs(gs, phi, debug):
    ''' Python implementation of IDEA-VB17/n4/pkg/MrServers/MrMeasSrv/SeqFW/libGSL/fGSLCalcPRS.cpp
    Calculates the phase encoding and readout direction vectors

    :param gs: The GS vector (= slice normal vector)
    :param phi: The rotational angle around Gs

    :return: gp: phase direction vector
    :return: gr: read direction vector
    '''

    # PCS axes
    SAGITTAL   = 0
    CORONAL    = 1
    TRANSVERSE = 2

    # Start of function
    orientation = 0  # will be one of SAGITTAL, CORONAL or TRANSVERSE (0, 1, or 2)
    orientation = orientation + class_ori(gs[SAGITTAL], gs[CORONAL], gs[TRANSVERSE], debug)
    gp  = np.zeros((3), dtype=float)

    if orientation == TRANSVERSE:
        gp[0] = 0.0
        gp[1] = gs[2] * np.sqrt(1. / (gs[1] * gs[1] + gs[2] * gs[2]))
        gp[2] = -gs[1] * np.sqrt(1. / (gs[1] * gs[1] + gs[2] * gs[2]))
    elif orientation == CORONAL:
        gp[0] = gs[1] * np.sqrt(1. / (gs[0] * gs[0] + gs[1] * gs[1]))
        gp[1] = -gs[0] * np.sqrt(1. / (gs[0] * gs[0] + gs[1] * gs[1]))
        gp[2] = 0.0
    elif orientation == SAGITTAL:
        gp[0] = -gs[1] * np.sqrt(1. / (gs[0] * gs[0] + gs[1] * gs[1]))
        gp[1] = gs[0] * np.sqrt(1. / (gs[0] * gs[0] + gs[1] * gs[1]))
        gp[2] = 0.0
    else:
        raise ValueError('Invalid slice orientation returned from class_ori')

    # Calculate GR = GS x GP
    gr = np.zeros((3), dtype=float)
    gr[0] = gs[1] * gp[2] - gs[2] * gp[1]
    gr[1] = gs[2] * gp[0] - gs[0] * gp[2]
    gr[2] = gs[0] * gp[1] - gs[1] * gp[0]

    if debug:
        print('Before rotation around S:')
        print(f'GP = {gp[0]:10.7f} {gp[1]:10.7f} {gp[2]:10.7f}')
        print(f'GR = {gr[0]:10.7f} {gr[1]:10.7f} {gr[2]:10.7f}')
        print(f'GS = {gs[0]:10.7f} {gs[1]:10.7f} {gs[2]:10.7f}')

    # rotation
    if phi != 0.0:
        # Rotate by phi around the S axis
        if debug:
            tmp = phi * 180.0 / np.pi
            print(f'PHI = {tmp:10.7f}')

    gp[0] = np.cos(phi) * gp[0] - np.sin(phi) * gr[0]
    gp[1] = np.cos(phi) * gp[1] - np.sin(phi) * gr[1]
    gp[2] = np.cos(phi) * gp[2] - np.sin(phi) * gr[2]

    # Calculate new GR = GS x GP
    gr[0] = gs[1] * gp[2] - gs[2] * gp[1]
    gr[1] = gs[2] * gp[0] - gs[0] * gp[2]
    gr[2] = gs[0] * gp[1] - gs[1] * gp[0]

    if debug:
        print('After the Rotation around S:')
        print(f'GP = {gp[0]:10.7f} {gp[1]:10.7f} {gp[2]:10.7f}')
        print(f'GR = {gr[0]:10.7f} {gr[1]:10.7f} {gr[2]:10.7f}')
        print(f'GS = {gs[0]:10.7f} {gs[1]:10.7f} {gs[2]:10.7f}')

    return gp, gr

def twix2DCMOrientation(mapVBVDHdr, force_svs=False, verbose=False):
    """ Convert twix orientation information to DICOM equivalent.

    Convert orientation to DICOM imageOrientationPatient, imagePositionPatient,
    pixelSpacing and sliceThickness field values.

    Args:
        mapVBVDHdr (dict): Header info interpreted by pymapVBVD
        force_svs (bool,optionl): Forces svs orientation information (size) to be read
        verbose (bool,optionl)
    Returns:
        imageOrientationPatient
        imagePositionPatient
        pixelSpacing
        sliceThickness

    """
    # Only single-slices are supported -- throw an error otherwise

    if ('sGroupArray', 'asGroup', '0', 'nSize') in mapVBVDHdr['MeasYaps']:
        nSlices = mapVBVDHdr['MeasYaps'][('sGroupArray', 'asGroup', '0', 'nSize')]
        if nSlices != 1.0:
            raise ValueError('In slice-selective spectroscopy, only the first slice is supported')

    # Orientation information
    # Added the force_svs because in some sequences there are slice objects initialised
    # and recorded but this seems sporadic behaviour.
    if ('sSliceArray', 'asSlice', '0', 'sNormal', 'dSag') in mapVBVDHdr['MeasYaps']\
            and not force_svs:
        # This is for slice-selective spectroscopy
        NormaldSag = mapVBVDHdr['MeasYaps'][('sSliceArray', 'asSlice', '0', 'sNormal', 'dSag')]
    elif ('sSpecPara', 'sVoI', 'sNormal', 'dSag') in mapVBVDHdr['MeasYaps']:
        NormaldSag = mapVBVDHdr['MeasYaps'][('sSpecPara', 'sVoI', 'sNormal', 'dSag')]
    else:
        NormaldSag = 0.0

    if ('sSliceArray', 'asSlice', '0', 'sNormal', 'dCor') in mapVBVDHdr['MeasYaps']\
            and not force_svs:
        NormaldCor = mapVBVDHdr['MeasYaps'][('sSliceArray', 'asSlice', '0', 'sNormal', 'dCor')]
    elif ('sSpecPara', 'sVoI', 'sNormal', 'dCor') in mapVBVDHdr['MeasYaps']:
        NormaldCor = mapVBVDHdr['MeasYaps'][('sSpecPara', 'sVoI', 'sNormal', 'dCor')]
    else:
        NormaldCor = 0.0

    if ('sSliceArray', 'asSlice', '0', 'sNormal', 'dTra') in mapVBVDHdr['MeasYaps']\
            and not force_svs:
        NormaldTra = mapVBVDHdr['MeasYaps'][('sSliceArray', 'asSlice', '0', 'sNormal', 'dTra')]
    elif ('sSpecPara', 'sVoI', 'sNormal', 'dTra') in mapVBVDHdr['MeasYaps']:
        NormaldTra = mapVBVDHdr['MeasYaps'][('sSpecPara', 'sVoI', 'sNormal', 'dTra')]
    else:
        NormaldTra = 0.0

    if ('sSliceArray', 'asSlice', '0', 'dInPlaneRot') in mapVBVDHdr['MeasYaps']\
            and not force_svs:
        inplaneRotation = mapVBVDHdr['MeasYaps'][('sSliceArray', 'asSlice', '0', 'dInPlaneRot')]
    elif ('sSpecPara', 'sVoI', 'dInPlaneRot') in mapVBVDHdr['MeasYaps']:
        inplaneRotation = mapVBVDHdr['MeasYaps'][('sSpecPara', 'sVoI', 'dInPlaneRot')]
    else:
        inplaneRotation = 0.0

    TwixSliceNormal = np.array([NormaldSag, NormaldCor, NormaldTra], dtype=float)
    # If all zeros make a 'normal' orientation (e.g. for unlocalised data)
    if not TwixSliceNormal.any():
        TwixSliceNormal[0] += 1.0

    if ('sSliceArray', 'asSlice', '0', 'dReadoutFOV') in mapVBVDHdr['MeasYaps']\
            and not force_svs:
        RoFoV = mapVBVDHdr['MeasYaps'][('sSliceArray', 'asSlice', '0', 'dReadoutFOV')]
        PeFoV = mapVBVDHdr['MeasYaps'][('sSliceArray', 'asSlice', '0', 'dPhaseFOV')]
    elif ('sSpecPara', 'sVoI', 'dReadoutFOV') in mapVBVDHdr['MeasYaps']:
        RoFoV = mapVBVDHdr['MeasYaps'][('sSpecPara', 'sVoI', 'dReadoutFOV')]
        PeFoV = mapVBVDHdr['MeasYaps'][('sSpecPara', 'sVoI', 'dPhaseFOV')]
    else:
        RoFoV = 10000.0
        PeFoV = 10000.0

    if ('sSliceArray', 'asSlice', '0', 'dThickness') in mapVBVDHdr['MeasYaps']\
            and not force_svs:
        sliceThickness = mapVBVDHdr['MeasYaps'][('sSliceArray', 'asSlice', '0', 'dThickness')]
    elif ('sSpecPara', 'sVoI', 'dThickness') in mapVBVDHdr['MeasYaps']:
        sliceThickness = mapVBVDHdr['MeasYaps'][('sSpecPara', 'sVoI', 'dThickness')]
    else:
        sliceThickness = 10000.0

    # Position info (including table position)
    if ('sSliceArray', 'asSlice', '0', 'sPosition', 'dSag') in mapVBVDHdr['MeasYaps']\
            and not force_svs:
        PosdSag = mapVBVDHdr['MeasYaps'][('sSliceArray', 'asSlice', '0', 'sPosition', 'dSag')]
    elif ('sSpecPara', 'sVoI', 'sPosition', 'dSag') in mapVBVDHdr['MeasYaps']:
        PosdSag = mapVBVDHdr['MeasYaps'][('sSpecPara', 'sVoI', 'sPosition', 'dSag')]
    else:
        PosdSag = 0.0

    if ('sSliceArray', 'asSlice', '0', 'sPosition', 'dCor') in mapVBVDHdr['MeasYaps']\
            and not force_svs:
        PosdCor = mapVBVDHdr['MeasYaps'][('sSliceArray', 'asSlice', '0', 'sPosition', 'dCor')]
    elif ('sSpecPara', 'sVoI', 'sPosition', 'dCor') in mapVBVDHdr['MeasYaps']:
        PosdCor = mapVBVDHdr['MeasYaps'][('sSpecPara', 'sVoI', 'sPosition', 'dCor')]
    else:
        PosdCor = 0.0

    if ('sSliceArray', 'asSlice', '0', 'sPosition', 'dTra') in mapVBVDHdr['MeasYaps']\
            and not force_svs:
        PosdTra = mapVBVDHdr['MeasYaps'][('sSliceArray', 'asSlice', '0', 'sPosition', 'dTra')]
    elif ('sSpecPara', 'sVoI', 'sPosition', 'dTra') in mapVBVDHdr['MeasYaps']:
        PosdTra = mapVBVDHdr['MeasYaps'][('sSpecPara', 'sVoI', 'sPosition', 'dTra')]
    else:
        PosdTra = 0.0

    if ('lScanRegionPosSag',) in mapVBVDHdr['MeasYaps']:
        PosdSag += mapVBVDHdr['MeasYaps'][('lScanRegionPosSag',)]
    if ('lScanRegionPosCor',) in mapVBVDHdr['MeasYaps']:
        PosdCor += mapVBVDHdr['MeasYaps'][('lScanRegionPosCor',)]
    if ('lScanRegionPosTra',) in mapVBVDHdr['MeasYaps']:
        PosdTra += mapVBVDHdr['MeasYaps'][('lScanRegionPosTra',)]

    
    dColVec_vector, dRowVec_vector = calc_prs(TwixSliceNormal, inplaneRotation, verbose)
    imageOrientationPatient = np.stack((dRowVec_vector, dColVec_vector), axis=0)

    pixelSpacing = np.array([PeFoV, RoFoV])  # [RoFoV PeFoV];

    imagePositionPatient = np.array([PosdSag, PosdCor, PosdTra], dtype=float)

    dim_swapped = False

    if verbose:
        print(f'imagePositionPatient is {imagePositionPatient.ravel()}')
        print(f'imageOrientationPatient is \n{imageOrientationPatient}')
        print(f'{imageOrientationPatient.ravel()}')
        print(f'pixelSpacing is {pixelSpacing}')

    return imageOrientationPatient, imagePositionPatient, pixelSpacing, sliceThickness, dim_swapped


def twix2DICOMMetaInfo(mapVBVDHdr, verbose=False):

    imageOrientationPatient, imagePositionPatient, _, _, _ = twix2DCMOrientation(mapVBVDHdr)

    RoFoV = mapVBVDHdr['MeasYaps'][('sSliceArray', 'asSlice', '0', 'dReadoutFOV')]
    PeFoV = mapVBVDHdr['MeasYaps'][('sSliceArray', 'asSlice', '0', 'dPhaseFOV')]
    sliceThickness = mapVBVDHdr['MeasYaps'][('sSliceArray', 'asSlice', '0', 'dThickness')]

    baseResolution = mapVBVDHdr['MeasYaps'][('sKSpace', 'lBaseResolution')]
    phaseEncodingLines = mapVBVDHdr['MeasYaps'][('sKSpace', 'lPhaseEncodingLines')]
    partitions = mapVBVDHdr['MeasYaps'][('sKSpace', 'lPartitions')]

    pixelSpacing = np.array([PeFoV / phaseEncodingLines, RoFoV / baseResolution, sliceThickness / partitions])
    imageSize = np.array([PeFoV, RoFoV, sliceThickness])

    imageOrigin = np.asarray([PeFoV, RoFoV, sliceThickness - pixelSpacing[2]]) / -2 + imagePositionPatient

    # print(f'imagePositionPatient is {imagePositionPatient.ravel()}')
    # print(f'imageOrientationPatient is {imageOrientationPatient}')
    # print(f'imageOrigin is {imageOrigin}')
    # print(f'pixelSpacing is {pixelSpacing}')

    dicom_metainfo = {
        'orientation': imageOrientationPatient.ravel(),
        'origin': imageOrigin,
        'spacing': pixelSpacing,
        'size': imageSize
    }

    return dicom_metainfo


if __name__ == "__main__":
    twixObj = mapvbvd.mapVBVD('./rawdata/mybrain-exp-20240403/Raw_data/meas_MID00210_FID14052_fme_pCASL_BL1800_PLD2000_5_reps.dat', quiet=(1 < 2))
    mapVBVDHdr = twixObj[1].hdr
    dicom_metainfo = twix2DICOMMetaInfo(mapVBVDHdr, verbose=False)

    print(f'imageOrientationPatient is {dicom_metainfo["orientation"]}')
    print(f'imageOrigin is {dicom_metainfo["origin"]}')
    print(f'pixelSpacing is {dicom_metainfo["spacing"]}')
    print(f'imageSize is {dicom_metainfo["size"]}')

