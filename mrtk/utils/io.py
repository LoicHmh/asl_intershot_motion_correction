import h5py
import numpy as np
from pathlib import Path
from fsl.data.image import Image
from typing import Literal, Tuple, Optional, Any

def save_data(save_path: str | Path, data_dict: dict[str, Any]) -> None:

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    assert save_path.suffix == '.h5', "The file name must end with .h5"

    with h5py.File(save_path, 'w') as f:
        dt = h5py.string_dtype(encoding='utf-8')
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                f.create_dataset(key, data=value)
            elif isinstance(value, Path):
                f.create_dataset(key, data=str(value), dtype=dt)
            elif isinstance(value, str):
                f.create_dataset(key, data=value, dtype=dt)
            elif value is None:
                f.create_dataset(key, data="__none__")
                f[key].attrs["__type__"] = "none"
            elif isinstance(value, list):
                f.create_dataset(key, data=np.array(value, dtype=int))
            elif isinstance(value, bool):
                f.create_dataset(key, data=int(value))
                f[key].attrs["__type__"] = "bool" 
            else:
                raise TypeError(f"Unsupported data type: {type(value)}")


def load_data(save_path: str | Path) -> dict[str, Any]:
    save_path = Path(save_path)
    assert save_path.exists(), f"File {save_path} does not exist"
    assert save_path.suffix == '.h5', "The file name must end with .h5"

    data_dict: dict[str, Any] = {}
    with h5py.File(save_path, 'r') as f:
        # for key in f.keys():
        #     if isinstance(f[key][()], bytes):
        #         data_dict[key] = f[key][()].decode('utf-8')
        #     else:
        #         data_dict[key] = f[key][()]
        # for key, value in f.attrs.items():
        #     if value == 'NoneType':
        #         data_dict[key] = None

        for key in f:
            ds = f[key]
            value = ds[()]
            tag = ds.attrs.get("__type__", None)
            if tag == "bool":
                value = bool(value)
            elif tag == "none":
                value = None
            else:
                if isinstance(value, bytes):
                    value = value.decode('utf-8')
            data_dict[key] = value

    return data_dict


def save_nii(img: np.ndarray, 
             save_path: str | Path, 
             header = None, 
             flip: Optional[Tuple[int]] = None, 
             roll: Optional[Tuple[int, int, int]] = None, 
             ) -> None:

    if img.ndim == 3:
        img = img[:, :, :, np.newaxis]

    if flip is not None:
        img = np.flip(img, axis=flip)

    if roll is not None:
        img = np.roll(img, shift=roll, axis=(0, 1, 2))

    if np.iscomplexobj(img):
        img = np.abs(img)
    
    img = img.astype(np.float32)    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image(img, header=header).save(save_path)
    print(f"Saved NIfTI image to {save_path}")


def add_suffix(path: Path, suffix: str) -> Path:
    """
    Add a suffix to the filename before the file extension.
    """
    full_suffix = ''.join(path.suffixes)
    stem = path.name[:-len(full_suffix)] if full_suffix else path.stem
    new_name = stem + suffix + full_suffix
    return path.with_name(new_name)


def save_asl(img4d: np.ndarray, 
             save_path: str | Path, 
             header = None, 
             perfusion_subtraction: Literal['magnitude', 'both', 'complex', 'none'] = 'both',
             flag_save_magnitude_value: bool = True,
             flag_save_phase_value: bool = False,
             flag_tag_first: bool = True,
             **kwargs) -> None:

    assert img4d.ndim == 4, "ASL Image must be 4D"
    assert img4d.shape[-1] % 2 == 0, "ASL Image must have an even number of time points"
 
    if flag_tag_first:
        ts, cs = 0, 1
    else:
        ts, cs = 1, 0

    if flag_save_magnitude_value:
        save_nii(np.abs(img4d), save_path=save_path, header=header, **kwargs)

    if flag_save_phase_value and np.iscomplexobj(img4d):
        save_nii(np.angle(img4d), save_path=add_suffix(save_path, '_phase'), header=header, **kwargs)

    if perfusion_subtraction == 'magnitude' or perfusion_subtraction == 'both':
        perf = np.abs(img4d[:, :, :, cs::2]) - np.abs(img4d[:, :, :, ts::2])
        save_nii(perf, save_path=add_suffix(save_path, '_perf_magn_sub'), header=header, **kwargs)

    if perfusion_subtraction == 'complex' or perfusion_subtraction == 'both':
        if np.iscomplexobj(img4d): 
            perf = np.abs(img4d[:, :, :, cs::2] - img4d[:, :, :, ts::2])
            save_nii(perf, save_path=add_suffix(save_path, '_perf_cplx_sub'), header=header, **kwargs)
        else:
            print("Warning: img4d is not complex, skipping complex subtraction.")
