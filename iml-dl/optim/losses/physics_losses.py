import torch
from mr_utils.parameter_fitting import T2StarFit


class ModelFitError:
    """
    Computes the ModelFitError as averaged residuals of a least squares fit
    (model: exponential decay) to the input data.
    """

    def __init__(self, mask_image=True, te1=5, d_te=5,
                 error_type='squared_residuals', perform_bgf_corr=False):
        super(ModelFitError, self).__init__()
        self.mask_image = mask_image
        self.te1 = te1
        self.d_te = d_te
        self.error_type = error_type
        self.perform_bgf_corr = perform_bgf_corr

    @staticmethod
    def _decay_model(t, t2star, a0, corr_factor=None):
        """ Exponential decay model. """

        if corr_factor is None:
            return a0 * torch.exp(-t / (t2star+1e-9))
        else:
            return a0 * torch.exp(-t / (t2star+1e-9)) / (corr_factor+1e-9)

    def _calc_fitted_signal(self, orig_signal, t2star, a0, corr_factor, echo_mask):
        """ Calculate the fitted signal intensities. """

        echo_times = torch.arange(self.te1, orig_signal.shape[0] * self.te1 + 1,
                                  self.d_te, dtype=orig_signal.dtype,
                                  device=orig_signal.device, requires_grad=True)
        echo_times = echo_times.unsqueeze(1).repeat(1, orig_signal.shape[1])
        t2star = t2star.unsqueeze(0).repeat(orig_signal.shape[0], 1)
        a0 = a0.unsqueeze(0).repeat(orig_signal.shape[0], 1)

        fit_signal =  self._decay_model(echo_times, t2star, a0, corr_factor)

        if echo_mask is not None:
            echo_mask[0:4] = 1

        return fit_signal, echo_mask

    def _calc_residuals(self, orig_signal, t2star, a0, corr_factor, echo_mask):
        """ Calculate the residuals of the least squares fit. """

        fit_signal, echo_mask = self._calc_fitted_signal(orig_signal, t2star, a0,
                                                         corr_factor, echo_mask)
        if echo_mask is not None:
            return orig_signal*echo_mask - fit_signal*echo_mask
        else:
            return orig_signal - fit_signal

    def _calc_emp_corr(self, orig_signal, t2star, a0, corr_factor, echo_mask):
        """
        Calculate the empirical correlation coefficient (= Pearson)
        between the original and the fitted signal intensities.

        Note: Values of the correlation coefficient are between -1 and 1.
        The error is calculated as 1-emp_corr to have a value
        between 0 and 1.
        """

        fit_signal, echo_mask = self._calc_fitted_signal(orig_signal, t2star,
                                                         a0, corr_factor,
                                                         echo_mask)
        if echo_mask is not None:
            fit_signal *= echo_mask
            orig_signal *= echo_mask

        mean_orig = torch.mean(orig_signal, dim=0)
        mean_fit = torch.mean(fit_signal, dim=0)

        numerator = torch.sum(
            (orig_signal - mean_orig.unsqueeze(0))
            * (fit_signal - mean_fit.unsqueeze(0)),
            dim=0
        )
        denominator = torch.sqrt(
            torch.sum((orig_signal - mean_orig.unsqueeze(0))**2, dim=0) *
            torch.sum((fit_signal - mean_fit.unsqueeze(0))**2, dim=0)
            + 1e-9
        )
        return 1 - torch.mean(numerator/(denominator+1e-9))

    def __call__(self, img, mask=None):
        """
        Calculate squared residuals of a least squares fit or
        empirical correlation coefficient as ModelFitError
        for input magnitude images (x) and image mask (mask).

        Note: currently, the T2* values are not clipped to a maximum value.

        Parameters
        ----------
        img : torch.Tensor
            Input magnitude images.
        mask : torch.Tensor
            Mask tensor indicating regions to include in the loss.

        Returns
        -------
        loss : torch.Tensor
            ModelFitError of the input magnitude images.
        """

        if self.mask_image and mask is None:
            print("ERROR: Masking is enabled but no mask is provided.")

        if not self.mask_image:
            mask = torch.ones_like(img)
        mask = mask[:, 0] > 0

        calc_t2star = T2StarFit(dim=4, te1=self.te1, d_te=self.d_te,
                                perform_bgf_correction=self.perform_bgf_corr,
                                mode="backprop")

        t2star, a0, corr_factor, echo_mask = calc_t2star(img)
        t2star, a0 = t2star[mask], a0[mask]
        t2star, a0 = t2star.flatten(), a0.flatten()
        if echo_mask is not None:
            echo_mask = echo_mask.permute(3, 0, 1, 2)[:, mask]
            corr_factor = corr_factor.permute(3, 0, 1, 2)[:, mask]
        data = torch.abs(img.permute(1, 0, 2, 3)[:, mask])

        if self.error_type == 'squared_residuals':
            residuals = self._calc_residuals(data, t2star, a0, corr_factor,
                                             echo_mask)
            error = torch.mean(residuals**2)
        elif self.error_type == 'absolute_residuals':
            residuals = self._calc_residuals(data, t2star, a0, corr_factor,
                                             echo_mask)
            error = torch.mean(torch.abs(residuals))
        elif self.error_type == 'emp_corr':
            error = self._calc_emp_corr(data, t2star, a0, corr_factor,
                                        echo_mask)
        else:
            raise ValueError('Invalid error type.')
        return error


class T2StarDiff:
    """
    Computes the T2StarDiff as mean absolute error between T2star maps
    resulting from predicted and ground truth images.

    Note: this class has been adapted to use the T2StarFit class for
    calculating T2* values. This still needs to be tested and validated.
    """

    def __init__(self, mask_image=True, te1=5, d_te=5, perform_bgf_corr=False):
        super(T2StarDiff, self).__init__()
        self.mask_image = mask_image
        self.te1 = te1
        self.d_te = d_te
        self.perform_bgf_corr = perform_bgf_corr

    def __call__(self, img_gt, img_pred, mask=None):
        """
        Calculate mean absolute error between T2star maps resulting from
        predicted and ground truth images.

        Note: currently, the T2* values are clipped between 0 and 200.

        Parameters
        ----------
        img_gt : torch.Tensor
            Input  ground truth images.
        img_pred : torch.Tensor
            Input predicted images.
        mask : torch.Tensor
            Mask tensor indicating regions to include in the loss.

        Returns
        -------
        loss : torch.Tensor
            T2StarDiff of the input ground truth and predicted images.
        """

        if self.mask_image and mask is None:
            print("ERROR: Masking is enabled but no mask is provided.")

        if not self.mask_image:
            mask = torch.ones_like(img_pred)
        mask = mask[:, 0] > 0

        calc_t2star = T2StarFit(dim=4, te1=self.te1, d_te=self.d_te,
                                perform_bgf_correction=self.perform_bgf_corr,
                                mode="backprop")

        t2star_gt, _, _, _ = calc_t2star(img_gt)
        t2star_pred, _, _, _ = calc_t2star(img_pred)
        t2star_gt, t2star_pred = t2star_gt[mask], t2star_pred[mask]

        return torch.mean(torch.abs(t2star_gt - t2star_pred))
