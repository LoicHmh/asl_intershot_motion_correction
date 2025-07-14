import mapvbvd
from mrtk.utils.twix2dicom_metainfo import twix2DICOMMetaInfo


def load_twix(args, verbose=1, squeeze=True, return_slice_order=False, logger=None, return_metainfo=False):
    if logger is not None:
        print_ = logger.debug
    else:
        print_ = print
    print_(f'Loading twix data from {args.raw_data_path}')

    twixObj = mapvbvd.mapVBVD(args.raw_data_path, quiet=(verbose < 2))
    if return_metainfo:
        dicom_metainfo = twix2DICOMMetaInfo(twixObj[1].hdr, verbose=False)
    im_twix = twixObj[1].image
    # im_twix = twixObj.image
    im_twix.flagRemoveOS = args.flagRemoveOS 
    im_twix.flagRampSampRegrid = args.flagRampSampRegrid
    im_twix.flagAverageReps = args.flagAverageReps
    im_twix.flagIgnoreSeg = args.flagIgnoreSeg

    im_array = im_twix[:].reshape(im_twix.dataSize)


    print_(f"Dimension:\t|" + "\t|  ".join(im_twix.dataDims))
    full_size = [f'{d:3d}' for d in im_twix.fullSize]

    print_(f"Full size:\t|" + "\t|  ".join(full_size))

    data_size = [f'{d:3d}' for d in im_twix.dataSize]
    if im_twix.flagRemoveOS:
        print_(f"Data size:\t|" + "\t|  ".join(data_size))

    par_order = []
    for i in im_twix.Par:
        if i not in par_order:
            par_order.append(int(i))

    if squeeze:
        im_array = im_array.squeeze()

        squeeze_size = [f'{d:3d}' if d>1 else '   ' for d in im_twix.dataSize]
        print_(f"Squeeze size:\t|" + "\t|  ".join(squeeze_size))

    returns = [im_array]
    if return_slice_order:
        returns.append(par_order)
    if return_metainfo:
        returns.append(dicom_metainfo)
    return returns

    
    

def load_twix_pc(args, verbose=1, squeeze=True, logger=None):
    if logger is not None:
        print_ = logger.debug
    else:
        print_ = print
    print_(f'Loading twix data (phasecor) from {args.raw_data_path}')
    twixObj = mapvbvd.mapVBVD(args.raw_data_path, quiet=(verbose < 2))
    pc_twix = twixObj[1].phasecor
    # pc_twix = twixObj.phasecor
    pc_twix.flagRemoveOS = args.flagRemoveOS 
    pc_twix.flagRampSampRegrid = args.flagRampSampRegrid
    pc_twix.flagSkipToFirstLine = True
    pc_twix.flagAverageReps = args.flagAverageReps
    pc_twix.flagIgnoreSeg = args.flagIgnoreSeg

    pc_array = pc_twix[:].reshape(pc_twix.dataSize)

    print_(f"Dimension:\t|" + "\t|  ".join(pc_twix.dataDims))
    full_size = [f'{d:3d}' for d in pc_twix.fullSize]
    print_(f"Full size:\t|" + "\t|  ".join(full_size))
    data_size = [f'{d:3d}' for d in pc_twix.dataSize]
    if pc_twix.flagRemoveOS:
        print_(f"Data size:\t|" + "\t|  ".join(data_size))
        
    if squeeze:
        squeeze_size = [f'{d:3d}' if d>1 else '   ' for d in pc_twix.dataSize]
        print_(f"Squeeze size:\t|" + "\t|  ".join(squeeze_size))
        return pc_array.squeeze()
    else:
        return pc_array