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
        frozen: bool = True,
    ) -> None:
        """Create an `AnchorPointEstimator` instance."""
        super().__init__(class_count)
        self.transitions = nn.Parameter(
            torch.empty((class_count, class_count)), requires_grad=not frozen
        )
        self.filter_outlier = True  # Remove after experimentation
        # Update the transition matrix using the multi-class anchor point method
        self.transition_matrix_from_anchors(classifier, sample_dataloader)

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
                this_noisy_posteriors = classifier(feats).cpu()
                if noisy_posteriors is None:
                    noisy_posteriors = this_noisy_posteriors
                else:
                    # Shape: (num_samples, class_count)
                    noisy_posteriors = torch.cat((noisy_posteriors, this_noisy_posteriors), dim=0)

            print(noisy_posteriors.shape)
            print(noisy_posteriors)
            for i in range(self.class_count):
                if self.filter_outlier:
                    # TODO: reference source code
                    eta_thresh = np.percentile(noisy_posteriors[:, i], 95, interpolation="higher")
                    robust_posteriors = noisy_posteriors
                    robust_posteriors[robust_posteriors >= eta_thresh] = 0.0
                    idx_best = torch.argmax(robust_posteriors[:, i])
                else:
                    idx_best = torch.argmax(noisy_posteriors[:, i])
                for j in range(self.class_count):
                    # TODO: check again whether this should be Tij or Tji
                    self.transitions[i, j] = noisy_posteriors[idx_best, j]

            # Row normalise
            row_sums = self.transitions.sum(axis=1)
            self.transitions /= row_sums[:, np.newaxis]

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Propagate data through the model.

        Params:
        -------
        `features`: input of shape (batch_size, class_count)

        Returns:
        --------
        `output`: output of shape (batch_size, class_count)
        """
        return torch.matmul(features, self.transitions)