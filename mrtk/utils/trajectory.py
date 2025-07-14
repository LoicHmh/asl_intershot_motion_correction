import numpy as np
# from deprecated import deprecated

# def sampling_caipi(shape, h, w, binary=False):
#     sampling_mask = np.zeros(shape, dtype=np.int8)
#     for h_i in range(h):
#         sampling_mask[:, h_i::h] += (h_i + 1)
#     for w_i in range(w):
#         sampling_mask[w_i::w, :] += (w_i * h)

#     for h_i in range(h):
#         for w_i in range(w):
#             sampling_mask[:, h_i + w_i * h::w*h] = np.roll(sampling_mask[:, h_i + w_i * h::w*h], w_i, axis=0)
    
#     if binary:
#         sampling_mask_bin = []
#         for i in range(h * w):
#             sampling_mask_bin.append(sampling_mask == (i + 1))

#         sampling_mask_bin = np.stack(sampling_mask_bin, axis=2).astype(np.int8)
#         return sampling_mask_bin
#     else:
#         return sampling_mask
    

def sampling_caipi(sy, sz, Ry, Rz, z_shift, shot=None):
    """
    $$ R_{CAIPI} = R_y \times R_z ^{(\Delta)} $$
    with $\Delta$ representing the applied shift in the $k_z$ direction
    from one sampling row $k_y$ to the next, and $R_z = R/R_y$.
    """

    R = Ry * Rz
    sampling_mask = np.zeros((R, R), dtype=np.int8)

    for y_i in range(Ry):
        sampling_mask[y_i::Ry, :] += (y_i * Rz)
    for z_i in range(Rz):
        sampling_mask[:, z_i::Rz] += (z_i + 1)

    for y_i in range(R):
        z_shift = (y_i // Ry) * z_shift % Rz
        sampling_mask[y_i, :] = np.roll(sampling_mask[y_i, :], z_shift, axis=0)

    sampling_mask = np.pad(sampling_mask, pad_width=((0, sy - R), (0, sz - R)), mode='wrap')

    if shot is None:
        return sampling_mask
    else:
        return (sampling_mask == shot).astype(np.int8)


def test_caipi():
    import matplotlib.pyplot as plt

    sy = 8
    sz = 8
    Ry = 4
    Rz = 2
    shift = 1
    shot = 1

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(sampling_caipi(sy=sy, sz=sz, Ry=Ry, Rz=Rz, shift=shift, shot=shot).T, cmap='gray')
    plt.title(f'CAIPI {sy=}, {sz=}, {Ry=}, {Rz=}, {shift=}, {shot=}')
    plt.ylabel('kz')
    plt.xlabel('ky')


    plt.subplot(1, 2, 2)
    plt.imshow(sampling_caipi(sy=sy, sz=sz, Ry=Ry, Rz=Rz, shift=shift).T, cmap='gray')
    plt.title(f'CAIPI {sy=}, {sz=}, {Ry=}, {Rz=}, {shift=}, {shot=}')
    plt.ylabel('kz')
    plt.xlabel('ky')
    plt.colorbar()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    test_caipi()
    
