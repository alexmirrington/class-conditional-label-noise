"""An implementation of Cross Entropy Loss with label smoothing."""
import torch.nn as nn


class LabelSmoothingCrossEntropyLoss(nn.Module):
    """Class to calculate cross entropy loss with label smoothing.

    Implementation predominantly based off wangleiofficial's contribution at
    https://github.com/wangleiofficial/label-smoothing-pytorch. The appropriate licence
    is included below.

    MIT License

    Copyright (c) 2020 wanglei

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """

    def __init__(self, smooth_val: float = 0.1, reduction="mean"):
        """Initialise `LabelSmoothingCrossEntropy` instance.

        Args
        ---
        `smooth_val`: If the target class is i, and e_i is the associated one-hot vector, then
                        the smoothed label will be e_i*(1-smooth_val) + smooth_val/num_classes*J,
                        where J is the all ones matrix.
        `reduction`: How to calculate total loss across batch.
        """
        super().__init__()
        self.smooth_val = smooth_val
        self.reduction = reduction

    def forward(self, preds, targets):
        """Return the cross entropy loss for the smoothed labels."""
        num_classes = preds.size()[-1]
        log_preds = nn.functional.log_softmax(preds, dim=-1)
        loss = self._reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = nn.functional.nll_loss(log_preds, targets, reduction=self.reduction)
        return self._linear_combination(loss / num_classes, nll, self.smooth_val)

    def _linear_combination(self, x, y, smooth_val):
        return smooth_val * x + (1 - smooth_val) * y

    def _reduce_loss(self, loss, reduction="mean"):
        return loss.mean() if reduction == "mean" else loss.sum() if reduction == "sum" else loss
