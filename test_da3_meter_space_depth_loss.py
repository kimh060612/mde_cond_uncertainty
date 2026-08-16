import torch

from model.loss_target import ssi_independent_da3_meter_space_depth_loss


def test_out_of_range_aligned_depth_does_not_explode_loss():
    prediction = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    candidate_gt = torch.tensor([[[[0.5, 0.6], [2.0, 3.0]]]])
    canonical_gt = torch.tensor([[[[0.1, 0.2], [2.0, 3.0]]]])

    loss = ssi_independent_da3_meter_space_depth_loss(
        prediction,
        prediction,
        candidate_gt,
        canonical_gt,
    )

    assert torch.isfinite(loss).all()
    assert loss.item() < 1.0


if __name__ == "__main__":
    test_out_of_range_aligned_depth_does_not_explode_loss()
