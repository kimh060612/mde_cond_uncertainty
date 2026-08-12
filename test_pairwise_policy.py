import ast
from pathlib import Path

import torch

from model.loss_fn import groupwise_pairwise_probit_loss
from utils.validator_camind import pairwise_policy_roc_data


def _wandb_metric_filter():
    source = Path("main_train_camera_error_uncertainty.py").read_text()
    tree = ast.parse(source)
    nodes = [
        node for node in tree.body
        if isinstance(node, (ast.Assign, ast.FunctionDef))
        and (
            isinstance(node, ast.FunctionDef)
            and node.name == "_wandb_validation_metrics"
            or isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "_REMOVED_WANDB_METRICS"
                for target in node.targets
            )
        )
    ]
    namespace = {}
    exec(compile(ast.Module(nodes, type_ignores=[]), "<wandb-filter>", "exec"), namespace)
    return namespace["_wandb_validation_metrics"], source


def main() -> None:
    mean = torch.tensor([[0.20, 0.19, 0.05]], requires_grad=True)
    variance = torch.full((1, 3), 0.01, requires_grad=True)
    actual = torch.tensor([[0.100, 0.102, 0.200]])
    loss = groupwise_pairwise_probit_loss(mean, variance, actual, 0.01, 3.0)
    std = (torch.tensor(0.02).sqrt() + 1e-6)
    expected = -torch.special.log_ndtr(torch.tensor([
        -(0.20 - 0.05 - 0.01) / std,
        -(0.19 - 0.05 - 0.01) / std,
    ])).mean()
    assert torch.allclose(loss, expected), "the <3% tie must not enter the loss mean"
    loss.backward()
    assert mean.grad is not None and variance.grad is None, "detached pair std leaked gradients"
    learned_variance = torch.full((1, 3), 0.01, requires_grad=True)
    groupwise_pairwise_probit_loss(
        mean.detach(), learned_variance, actual, 0.01, 3.0, detach_pair_std=False
    ).backward()
    assert learned_variance.grad is not None, "detach_pair_std=false must train pair std"
    assert groupwise_pairwise_probit_loss(
        mean[:, :2], variance[:, :2], actual[:, :2], 0.01, 3.0
    ).item() == 0.0, "tie-only groups must contribute zero"

    two_group_mean = torch.tensor([[0.3, 0.2, 0.1], [0.2, 0.19, 0.05]])
    two_group_var = torch.full_like(two_group_mean, 0.01)
    two_group_actual = torch.tensor([[0.1, 0.2, 0.4], [0.1, 0.102, 0.2]])
    combined = groupwise_pairwise_probit_loss(
        two_group_mean, two_group_var, two_group_actual, 0.01, 3.0
    )
    separate = torch.stack([
        groupwise_pairwise_probit_loss(
            two_group_mean[index:index + 1], two_group_var[index:index + 1],
            two_group_actual[index:index + 1], 0.01, 3.0,
        )
        for index in range(2)
    ]).mean()
    assert torch.allclose(combined, separate), "groups must be weighted equally"

    camera_mean = torch.tensor([0.20, 0.10, 0.12])
    camera_std = torch.tensor([0.05, 0.05, 0.05])
    candidate_abs_rel = torch.tensor([0.20, 0.10, 0.102])
    group_id = torch.zeros(3)
    low_beta = pairwise_policy_roc_data(
        camera_mean, camera_std, candidate_abs_rel, group_id, [3.0], 0.01, 0.0
    )[3.0]
    high_beta = pairwise_policy_roc_data(
        camera_mean, camera_std, candidate_abs_rel, group_id, [3.0], 0.01, 3.0
    )[3.0]
    assert low_beta["num_pairs"] == 4, "one tie must remove both ordered directions"
    assert low_beta["labels"].tolist()[:2] == [True, True]
    assert low_beta["labels"].tolist()[2:] == [False, False]
    assert (low_beta["scores"][:2] > 0).all() and (low_beta["scores"][2:] < 0).all()
    assert torch.equal(low_beta["decisions"], low_beta["labels"])
    assert low_beta["recall"] > high_beta["recall"]
    assert low_beta["num_switches"] > high_beta["num_switches"]

    filter_metrics, source = _wandb_metric_filter()
    payload = {
        **filter_metrics("val_seen", {
            "loss": 1.0,
            "q_rank_accuracy": 0.8,
            "selection_accuracy": 0.5,
            "selection_accuracy_within_3pct": 0.7,
            "selection_mean_regret_abs_rel": 0.1,
            "groupwise_top1_selection_accuracy": 0.5,
            "mean_selected_regret": 0.1,
            "selection_alpha_sweep": object(),
        }),
        **filter_metrics("val_unseen", {"loss": 2.0, "q_vs_abs_rel_degradation_pearson": 0.4}),
        "val_seen/pairwise_policy_roc": object(),
        "val_unseen/pairwise_policy_roc": object(),
    }
    assert "val_seen/loss" in payload and "val_seen/q_rank_accuracy" in payload
    assert "val_unseen/loss" in payload and "val_unseen/q_vs_abs_rel_degradation_pearson" in payload
    assert set(key for key in payload if key.endswith("pairwise_policy_roc")) == {
        "val_seen/pairwise_policy_roc", "val_unseen/pairwise_policy_roc"
    }
    assert not any(
        token in key
        for key in payload
        for token in (
            "selection_accuracy", "selection_mean_regret_abs_rel",
            "groupwise_top1_selection_accuracy", "mean_selected_regret",
            "selection_alpha_sweep",
        )
    )
    assert '"val/pairwise_policy_roc"' not in source


if __name__ == "__main__":
    main()
