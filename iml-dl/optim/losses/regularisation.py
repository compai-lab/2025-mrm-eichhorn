import torch
import logging
import numpy as np


class EnforceDiverseDCWeights:
    """ Regulariser to enforce diverse DC weights for different
    acceleration rates. """

    def __init__(self):
        super(EnforceDiverseDCWeights, self).__init__()

    def __call__(self, model, device, min_acc=0.05, max_acc=0.5):
        """
        Compute the difference between the predicted weights of the
        DC component for different exclusion rates.

        Parameters
        ----------
        weights : torch.Tensor
            Weights tensor.

        Returns
        -------
        loss : torch.Tensor
            EnforceDiverseDCWeights loss for the given weights.
        """

        exclusion_rates = sorted([np.random.uniform(min_acc, max_acc)
                                  for _ in range(12)])
        dc_weights = []
        for e in exclusion_rates:
            excl_rate = torch.tensor([e], device=device).to(
                torch.float32)
            dc_weights.append([
                model.predict_parameters(hyper_input=excl_rate)[k]
                for k in model.predict_parameters(hyper_input=excl_rate).keys()
                if 'DC' in k
            ])

        dc_weights = torch.stack([
            torch.stack(sublist) for sublist in dc_weights
        ])
        squared_diff = torch.sum(
            torch.abs(dc_weights[0:6] - dc_weights[6:]) ** 2,
            dim=1
        )
        max_weight = torch.amax(torch.cat([
            torch.abs(dc_weights[0:6]) ** 2,
            torch.abs(dc_weights[6:]) ** 2], dim=1), dim=1)

        return 1 - torch.mean(squared_diff / (max_weight * dc_weights.shape[1]))


class EnforcedSmallExclRates:
    """ Regulariser to enforce small exclusion rates. """

    def __init__(self, scaling_factor=0.5):
        super(EnforcedSmallExclRates, self).__init__()

        self.scaling_factor = scaling_factor

    def __call__(self, mask):

        centered_mask = 10 * (mask - 0.5)
        sigmoid_term = torch.sigmoid(-1 * centered_mask).sum() / mask.numel()
        mean_term = torch.mean(1 - torch.sum(mask, dim=1) / mask.shape[1])

        return (1 - self.scaling_factor) * sigmoid_term + self.scaling_factor * mean_term


class SmallExclRatesBinaryMask:
    """ Regulariser to enforce small exclusion rates. """

    def __init__(self, scaling_factor=0.5):
        super(SmallExclRatesBinaryMask, self).__init__()

        self.scaling_factor = scaling_factor

    def __call__(self, mask):

        centered_mask = 10 * (mask - 0.5)
        sigmoid_term = torch.sigmoid(-1 * centered_mask).sum() / mask.numel()
        binary_mask = mask > 0.5
        mean_term = torch.mean(binary_mask*(1-mask) + (~binary_mask)*mask)

        return (1 - self.scaling_factor) * sigmoid_term + self.scaling_factor * mean_term


class MeanExclusionRate:
    """ Regulariser to enforce small exclusion rates. """

    def __init__(self):
        super(MeanExclusionRate, self).__init__()

    def __call__(self, mask):
        return torch.mean(1 - torch.sum(mask, dim=1) / mask.shape[1])


class CenterWeightedMeanExclusionRate:
    """ Regulariser to enforce small exclusion rates. """

    def __init__(self, scaling_factor=1):
        super(CenterWeightedMeanExclusionRate, self).__init__()
        self.scaling_factor = scaling_factor

    def __call__(self, mask):
        center = mask.shape[1] // 2
        return torch.mean(
            (1 - torch.sum(mask, dim=1) / mask.shape[1])
            + self.scaling_factor * (1 - torch.sum(mask[:, center-5:center+5],
                                                   dim=1) / 10)
        )


class WeightedMeanExclusionRate:
    """ Regulariser to enforce small exclusion rates, with more weight in the
    center. """

    def __init__(self):
        super(WeightedMeanExclusionRate, self).__init__()

    def __call__(self, mask):
        distance_from_center = torch.abs(
            torch.arange(mask.shape[1]) - (mask.shape[1] - 1) / 2)
        weights = torch.exp(
            -0.5 * (distance_from_center ** 2) / (40 ** 2)
        ).to(mask.device)
        return torch.mean((1 - mask) * weights[None])


class ThreshMeanExclusionRate:
    """ Regulariser to enforce values close to 1. """

    def __init__(self, scaling_factor=0.5):
        super(ThreshMeanExclusionRate, self).__init__()

    def __call__(self, mask):
        centered_mask = 10 * (mask - 0.5)
        return torch.sigmoid(-1 * centered_mask).sum() / mask.numel()


class BinaryMask:
    """ Regulariser to enforce binary masks (values close to 0 or 1). """

    def __init__(self, scaling_factor=0.5):
        super(BinaryMask, self).__init__()

    def __call__(self, mask):
        binary_mask = mask > 0.5
        return torch.mean(binary_mask*(1-mask) + (~binary_mask)*mask)


class MaskVariabilityAcrossSlices:
    """  Regulariser to penalise variability of the masks across
    adjacent slices.
    """

    def __init__(self):
        super(MaskVariabilityAcrossSlices, self).__init__()

    @staticmethod
    def _sort_slices(mask, slice_num):
        """ Re-order the slices according to acquisition scheme. """

        even = slice_num % 2 == 0
        even_slices = slice_num[even]
        even_slices, even_indices = even_slices.sort(dim=0)

        odd = slice_num % 2 == 1
        odd_slices = slice_num[odd]
        odd_slices, odd_indices = odd_slices.sort(dim=0)

        slice_num_sorted = torch.cat((even_slices, odd_slices), dim=0).unsqueeze(1)
        mask_sorted = torch.cat((mask[even.squeeze()][even_indices],
                                 mask[odd.squeeze()][odd_indices]), dim=0)

        return mask_sorted, slice_num_sorted

    def __call__(self, mask, slice_num):
        """
        Compute the agreement between masks of adjacent slices,
        using sum of absolute differences.

        Note: According to the interleaved multi-slice acquisition scheme,
        first all even slices are acquired, followed by all odd slices,
        so the difference between adjacent slices is 2.

        Parameters
        ----------
        mask : torch.Tensor
            Mask tensor.
        slice_num : torch.Tensor
            Slice number tensor.

        Returns
        -------
        loss : torch.Tensor
            MaskVariabilityAcrossSlices loss for the given mask.
        """

        if slice_num.shape[0] > 1:
            mask, slice_num = self._sort_slices(mask, slice_num)

            slice_diff_next = slice_num[1:] - slice_num[:-1]
            adj_slices = abs(slice_diff_next) == 2
            adj_slices = torch.cat((
                adj_slices, torch.tensor([[False]]).to(adj_slices.device)
            ))
            abs_diff = torch.sum(torch.abs(
                mask[adj_slices.expand_as(mask)]
                - mask[torch.roll(adj_slices, shifts=1).expand_as(mask)]
            ))
            loss_reg = abs_diff / (torch.sum(adj_slices)*mask.shape[1])
            if torch.isnan(loss_reg):
                return None
            else:
                return loss_reg
        else:
            logging.info(
                "[Trainer::train]: Regularisation on mask variation "
                "can only be calculated for batch sizes > 1")
            return None


class MaskVariabilityAcrossSlicesMoreNeighbors:
    """  Regulariser to penalise variability of the masks across
    adjacent slices.
    """

    def __init__(self, nr_neighbors=1):
        super(MaskVariabilityAcrossSlicesMoreNeighbors, self).__init__()
        self.nr_neighbors = nr_neighbors


    @staticmethod
    def _sort_slices(mask, slice_num):
        """ Re-order the slices according to acquisition scheme. """

        even = slice_num % 2 == 0
        even_slices = slice_num[even]
        even_slices, even_indices = even_slices.sort(dim=0)

        odd = slice_num % 2 == 1
        odd_slices = slice_num[odd]
        odd_slices, odd_indices = odd_slices.sort(dim=0)

        slice_num_sorted = torch.cat((even_slices, odd_slices), dim=0).unsqueeze(1)
        mask_sorted = torch.cat((mask[even.squeeze()][even_indices],
                                 mask[odd.squeeze()][odd_indices]), dim=0)

        return mask_sorted, slice_num_sorted

    def __call__(self, mask, slice_num):
        """
        Compute the agreement between masks of adjacent slices,
        using sum of absolute differences.

        Note: According to the interleaved multi-slice acquisition scheme,
        first all even slices are acquired, followed by all odd slices,
        so the difference between adjacent slices is 2.

        Parameters
        ----------
        mask : torch.Tensor
            Mask tensor.
        slice_num : torch.Tensor
            Slice number tensor.

        Returns
        -------
        loss : torch.Tensor
            MaskVariabilityAcrossSlices loss for the given mask.
        """

        if slice_num.shape[0] > 1:
            mask, slice_num = self._sort_slices(mask, slice_num)
            total_abs_diff = 0
            count = 0

            for diff in range(2, 2*self.nr_neighbors+1, 2):
                slice_diff_next = slice_num[diff//2:] - slice_num[:-diff//2]
                adj_slices = abs(slice_diff_next) == diff
                adj_slices = torch.cat((
                    adj_slices, torch.tensor([[False]] * (diff//2)).to(adj_slices.device)
                ))
                abs_diff = torch.sum(torch.abs(
                    mask[adj_slices.expand_as(mask)]
                    - mask[torch.roll(adj_slices, shifts=diff//2).expand_as(mask)]
                ))
                total_abs_diff += abs_diff
                count += torch.sum(adj_slices)

            loss_reg = total_abs_diff / (count * mask.shape[1])

            if torch.isnan(loss_reg):
                return None
            else:
                return loss_reg
        else:
            logging.info(
                "[Trainer::train]: Regularisation on mask variation "
                "can only be calculated for batch sizes > 1")
            return None