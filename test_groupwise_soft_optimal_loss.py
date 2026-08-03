import torch
import torch.nn.functional as F

from model.loss_fn import groupwise_soft_optimal_loss


def main() -> None:
    bias = torch.tensor(
        [
            [0.4, 0.1, 0.3, 0.2, 0.8, 0.7, 0.6, 0.5],
            [0.8, 0.6, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1],
        ],
        requires_grad=True,
    )
    degradation = torch.tensor(
        [
            [0.4, 0.1, 0.3, 0.2, 0.8, 0.7, 0.6, 0.5],
            [0.7, 0.6, 0.8, 0.5, 0.4, 0.3, 0.2, 0.1],
        ],
        requires_grad=True,
    )
    target_temperature = 0.3
    prediction_temperature = 0.2

    loss = groupwise_soft_optimal_loss(
        bias,
        degradation,
        target_temperature,
        prediction_temperature,
    )
    relative_degradation = degradation - degradation.min(dim=1, keepdim=True).values
    target = F.softmax(-relative_degradation / target_temperature, dim=1).detach()
    expected = -(
        target * F.log_softmax(-bias / prediction_temperature, dim=1)
    ).sum(dim=1).mean()
    assert torch.allclose(loss, expected)
    assert torch.allclose(
        loss,
        groupwise_soft_optimal_loss(
            bias,
            degradation + torch.tensor([[7.0], [-3.0]]),
            target_temperature,
            prediction_temperature,
        ),
    )
    permutation = torch.tensor([5, 2, 7, 0, 3, 1, 6, 4])
    assert torch.allclose(
        loss,
        groupwise_soft_optimal_loss(
            bias[:, permutation],
            degradation[:, permutation],
            target_temperature,
            prediction_temperature,
        ),
    )

    loss.backward()
    assert bias.grad is not None
    assert degradation.grad is None


if __name__ == "__main__":
    main()
