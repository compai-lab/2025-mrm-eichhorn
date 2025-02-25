import merlinth
import torch
import merlinth

def process_kspace_linedet(img_cc_fs, sens_maps, coils_channel_dim=True,
                           coil_combined=False):
    """
    Process the k-space data for line detection from the input images and
    sensitivity maps.

    Parameters
    ----------
    img_cc_fs : torch.Tensor
        The input images.
    sens_maps : torch.Tensor
        The sensitivity maps.

    Returns
    -------
    torch.Tensor
        The processed k-space data.
    """

    if not coil_combined:
        coil_imgs = img_cc_fs.unsqueeze(2) * sens_maps
    else:
        coil_imgs = img_cc_fs.unsqueeze(2)
    kspace = merlinth.layers.mri.fft2c(coil_imgs)

    # Normalize each k-space line with the norm of the line
    # (all echoes and coils together)
    norm = torch.sqrt(torch.sum(torch.abs(kspace) ** 2, dim=(1, 2, 4), keepdim=True))
    kspace = kspace / norm

    # Split up the k-space data into real and imaginary parts and stack them
    # along the channel dimension
    if not coil_combined and not coils_channel_dim:
        kspace = kspace.reshape(kspace.shape[0], kspace.shape[1] * kspace.shape[2],
                                kspace.shape[3], kspace.shape[4])[:, None]
        kspace = torch.cat((kspace.real, kspace.imag), dim=1)
    else:
        kspace = torch.cat((kspace.real, kspace.imag), dim=2)
        kspace = kspace.permute(0, 2, 1, 3, 4)

    return kspace.to(torch.float64)
