"""Implementation of a forward-method transition matrix estimator."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .base import AbstractEstimator


class AnchorPointEstimator(AbstractEstimator):
    """A transition matrix estimator using anchor points from classifier trained on noisy data."""

    def __init__(
        self,
        classifier: nn.Module,
        sample_dataloader: DataLoader,
        class_count: int,
        outlier_percentile: float,
        frozen: bool = True,
    ) -> None:
        """Create an `AnchorPointEstimator` instance."""
        super().__init__(class_count)
        self.transitions = torch.empty((class_count, class_count))

        self.outlier_percentile = outlier_percentile * 100
        # Update the transition matrix using the multi-class anchor point method
        self.transition_matrix_from_anchors(classifier, sample_dataloader)
        self.inverse_transitions = torch.inverse(self.transitions)

    def transition_matrix_from_anchors(
        self, classifier: nn.Module, sample_dataloader: DataLoader
    ) -> None:
        """Extract transition matrix using the anchor point method.

        As detailed for the multi-class case in "Making Deep Neural Networks Robust to Label Noise:
        a Loss Correction Approach" [Patrini et al.], this involves taking the predictions of a
        classifier trained on the noisy data for data examples, and for each class, finding the
        sample for which that classifier is most confident the sample belongs to that class (this
        is the anchor point/sample for that class). Suppose x^i is the anchor point for the ith
        class. Then the transition matrix T is constructed as T_ij = p(y=j | x^i), where p is the
        probability output by the classifier trained on noisy data. For more details, please
        reference our accompanying report.

        Args
        ---
        classifier: a classifier trained on the noisy data to produce posteriors over the classes
        sample_dataloader: samples from which to choose the anchor points
                            (for example, but not necessarily, the training data)
        """
        classifier.eval()
        with torch.no_grad():
            noisy_posteriors = None
            # get matrix of probabilities for each example
            for feats, _ in sample_dataloader:
                this_noisy_posteriors, _ = classifier(feats)
                this_noisy_posteriors = this_noisy_posteriors.cpu()
                if noisy_posteriors is None:
                    noisy_posteriors = this_noisy_posteriors
                else:
                    # Shape: (num_samples, class_count)
                    noisy_posteriors = torch.cat((noisy_posteriors, this_noisy_posteriors), dim=0)

            for i in range(self.class_count):
                if self.outlier_percentile > 0 and self.outlier_percentile < 100:
                    # TODO: reference source code
                    eta_thresh = np.percentile(
                        noisy_posteriors[:, i], self.outlier_percentile, interpolation="higher"
                    )
                    robust_posteriors = noisy_posteriors[
                        torch.where(noisy_posteriors[:, i] < eta_thresh)
                    ]

                    idx_best = torch.argmax(robust_posteriors[:, i])
                    anchor_point = robust_posteriors[idx_best]
                else:
                    anchor_point = noisy_posteriors[torch.argmax(noisy_posteriors[:, i])]

                self.transitions[i, :] = anchor_point
