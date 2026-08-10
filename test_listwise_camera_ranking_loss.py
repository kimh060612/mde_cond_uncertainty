import torch
import torch.nn.functional as F

from model.loss_fn import listwise_camera_ranking_loss


def test_listwise_camera_ranking_loss_uses_one_global_list():
    predicted = torch.tensor([[0.2, -0.1], [0.4, 0.3]], requires_grad=True)
    target = torch.tensor([[0.3, 0.1], [0.2, 0.4]])
    temperature = 0.5

    loss = listwise_camera_ranking_loss(predicted, target, temperature)
    expected = -(
        F.softmax(-target.flatten() / temperature, dim=0)
        * F.log_softmax(-predicted.flatten() / temperature, dim=0)
    ).sum()

    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(predicted.grad).all()


if __name__ == "__main__":
    test_listwise_camera_ranking_loss_uses_one_global_list()
