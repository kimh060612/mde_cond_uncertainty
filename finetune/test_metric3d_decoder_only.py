import torch

from finetune.utils import metric3d_decoder_only, predict_metric3d_decoder_only


class Scale(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))


class Encoder(Scale):
    def forward(self, value):
        return value.mean(1, keepdim=True) * self.weight


class Decoder(Scale):
    def forward(self, features):
        return {"prediction": features * self.weight}


def main() -> None:
    model = torch.nn.Module()
    model.depth_model = torch.nn.Module()
    model.depth_model.encoder = Encoder()
    model.depth_model.decoder = Decoder()
    decoder = metric3d_decoder_only(model)

    model.train()  # run_epoch does this before each training epoch.
    prediction = predict_metric3d_decoder_only(
        model, torch.rand(2, 3, 32, 48), (32, 48), 1000.0
    )
    prediction.mean().backward()

    assert model.depth_model.encoder.weight.grad is None
    assert not model.depth_model.encoder.weight.requires_grad
    assert not model.depth_model.encoder.training
    assert decoder.weight.grad is not None
    assert decoder.weight.requires_grad


if __name__ == "__main__":
    main()
