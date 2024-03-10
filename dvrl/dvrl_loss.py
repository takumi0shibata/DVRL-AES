"""Loss function for data valuation"""

import torch
import torch.nn as nn


class DvrlLoss(nn.Module):
    def __init__(self, epsilon: float, threshold: float) -> None:
        """
        Construct class
        Args:
            epsilon: Small value to avoid overflow
            threshold: Encourages exploration
        """
        super().__init__()
        self.epsilon = epsilon
        self.threshold = threshold

    def forward(self, est_data_value, s_input, reward_input):
        """
        Calculate the loss.
        Args:
            est_data_value: Estimated data value
            s_input: data selection array
            reward_input: Reward
        Returns:
            dve_loss: Loss value
        """
        # Generator loss (REINFORCE algorithm)
        one = torch.ones_like(est_data_value, dtype=est_data_value.dtype)
        prob = torch.sum(s_input * torch.log(est_data_value + self.epsilon) + \
                         (one - s_input) * \
                         torch.log(one - est_data_value + self.epsilon))

        zero = torch.Tensor([0.0])
        zero = zero.to(est_data_value.device)

        print("prob: ", prob)
        print("reward_input: ", reward_input)
        dve_loss = (-reward_input * prob) + \
                   1e3 * torch.maximum(torch.mean(est_data_value) - self.threshold, zero) + \
                   1e3 * torch.maximum(1 - self.threshold - torch.mean(est_data_value), zero)

        return dve_loss
