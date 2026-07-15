import pytest
import torch

from model.hybrid_lm import (
    HybridLlamaSemanticLM,
    HybridSemanticLM,
    _semantic_loss_metrics,
)


def _loss_metrics(
    logits: torch.Tensor,
    targets: torch.LongTensor,
    target_mask: torch.BoolTensor,
    **overrides,
) -> dict[str, torch.Tensor]:
    kwargs = {
        "label_smoothing": 0.0,
        "ignore_index": 99,
        "transition_loss_weight": 1.0,
    }
    kwargs.update(overrides)
    return _semantic_loss_metrics(logits, targets, target_mask, **kwargs)


def test_transition_predecessor_margin_is_exact_and_ignores_padding():
    logits = torch.zeros(1, 6, 4)
    targets = torch.tensor([[0, 0, 1, 1, 2, 99]])
    target_mask = torch.tensor([[True, True, True, True, True, False]])
    logits[0, 2, 0] = 0.7
    logits[0, 2, 1] = 0.2
    logits[0, 4, 1] = 0.0
    logits[0, 4, 2] = 2.0

    metrics = _loss_metrics(
        logits,
        targets,
        target_mask,
        transition_loss_weight=3.0,
        transition_predecessor_margin=1.0,
        transition_predecessor_margin_weight=2.0,
    )

    torch.testing.assert_close(
        metrics["transition_predecessor_margin_loss"],
        torch.tensor(0.75),
    )
    torch.testing.assert_close(
        metrics["transition_predecessor_rate"],
        torch.tensor(0.5),
    )
    torch.testing.assert_close(
        metrics["objective_loss"],
        metrics["ce_objective_loss"] + 1.5,
    )


def test_transition_predecessor_margin_zero_transition_batch_is_finite():
    torch.manual_seed(101)
    logits = torch.randn(2, 4, 3, requires_grad=True)
    targets = torch.tensor([[1, 1, 1, 1], [2, 2, 99, 99]])
    target_mask = torch.tensor(
        [[True, True, True, True], [True, True, False, False]]
    )

    metrics = _loss_metrics(
        logits,
        targets,
        target_mask,
        transition_predecessor_margin_weight=4.0,
    )
    metrics["objective_loss"].backward()

    assert metrics["transition_predecessor_margin_loss"].item() == 0.0
    assert metrics["transition_predecessor_rate"].item() == 0.0
    assert metrics["objective_loss"] is metrics["ce_objective_loss"] or torch.equal(
        metrics["objective_loss"], metrics["ce_objective_loss"]
    )
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.equal(logits.grad[1, 2:], torch.zeros_like(logits.grad[1, 2:]))


def test_zero_margin_weight_is_exactly_backward_compatible():
    torch.manual_seed(102)
    logits = torch.randn(2, 5, 4)
    targets = torch.tensor([[0, 0, 1, 1, 2], [3, 2, 2, 99, 99]])
    target_mask = torch.tensor(
        [[True, True, True, True, True], [True, True, True, False, False]]
    )

    default = _loss_metrics(
        logits,
        targets,
        target_mask,
        transition_loss_weight=3.0,
        normalize_transition_weights_per_sample=True,
    )
    explicit_zero = _loss_metrics(
        logits,
        targets,
        target_mask,
        transition_loss_weight=3.0,
        normalize_transition_weights_per_sample=True,
        transition_predecessor_margin=7.0,
        transition_predecessor_margin_weight=0.0,
    )

    for key in default:
        if key == "transition_predecessor_margin_loss":
            continue
        assert torch.equal(default[key], explicit_zero[key])
    assert default["objective_loss"] is default["ce_objective_loss"]
    assert explicit_zero["objective_loss"] is explicit_zero["ce_objective_loss"]


def test_margin_uses_raw_logits_independently_of_label_smoothing():
    torch.manual_seed(103)
    logits = torch.randn(1, 5, 4)
    targets = torch.tensor([[0, 1, 1, 2, 3]])
    target_mask = torch.ones_like(targets, dtype=torch.bool)

    unsmoothed = _loss_metrics(
        logits,
        targets,
        target_mask,
        label_smoothing=0.0,
        transition_predecessor_margin_weight=2.0,
    )
    smoothed = _loss_metrics(
        logits,
        targets,
        target_mask,
        label_smoothing=0.2,
        transition_predecessor_margin_weight=2.0,
    )

    assert torch.equal(
        unsmoothed["transition_predecessor_margin_loss"],
        smoothed["transition_predecessor_margin_loss"],
    )
    torch.testing.assert_close(
        smoothed["objective_loss"] - smoothed["ce_objective_loss"],
        unsmoothed["objective_loss"] - unsmoothed["ce_objective_loss"],
    )


def test_margin_is_independent_of_transition_ce_weighting_and_normalization():
    torch.manual_seed(105)
    logits = torch.randn(2, 5, 4)
    targets = torch.tensor([[0, 0, 1, 2, 2], [3, 2, 2, 99, 99]])
    target_mask = torch.tensor(
        [[True, True, True, True, True], [True, True, True, False, False]]
    )
    margin_losses = []

    for transition_loss_weight in (1.0, 4.0):
        for normalize in (False, True):
            metrics = _loss_metrics(
                logits,
                targets,
                target_mask,
                transition_loss_weight=transition_loss_weight,
                normalize_transition_weights_per_sample=normalize,
                transition_predecessor_margin=1.5,
                transition_predecessor_margin_weight=0.25,
            )
            margin_losses.append(metrics["transition_predecessor_margin_loss"])

    for margin_loss in margin_losses[1:]:
        assert torch.equal(margin_loss, margin_losses[0])


def test_margin_adds_only_target_and_predecessor_logit_gradients():
    targets = torch.tensor([[0, 1]])
    target_mask = torch.ones_like(targets, dtype=torch.bool)
    baseline_logits = torch.zeros(1, 2, 3, requires_grad=True)
    margin_logits = baseline_logits.detach().clone().requires_grad_(True)

    baseline = _loss_metrics(
        baseline_logits,
        targets,
        target_mask,
        transition_predecessor_margin_weight=0.0,
    )
    treatment = _loss_metrics(
        margin_logits,
        targets,
        target_mask,
        transition_predecessor_margin_weight=1.0,
    )
    baseline["objective_loss"].backward()
    treatment["objective_loss"].backward()

    gradient_delta = margin_logits.grad - baseline_logits.grad
    expected = torch.zeros_like(gradient_delta)
    expected[0, 1, 0] = 1.0
    expected[0, 1, 1] = -1.0
    torch.testing.assert_close(gradient_delta, expected)


def test_margin_accumulates_in_float32_for_bfloat16_logits():
    logits = torch.zeros(1, 257, 3, dtype=torch.bfloat16, requires_grad=True)
    targets = torch.arange(257).remainder(2).unsqueeze(0)
    target_mask = torch.ones_like(targets, dtype=torch.bool)

    metrics = _loss_metrics(
        logits,
        targets,
        target_mask,
        transition_predecessor_margin=1.0,
        transition_predecessor_margin_weight=0.25,
    )
    metrics["objective_loss"].backward()

    assert metrics["transition_predecessor_margin_loss"].dtype == torch.float32
    torch.testing.assert_close(
        metrics["transition_predecessor_margin_loss"],
        torch.tensor(1.0),
    )
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("lm_class", [HybridSemanticLM, HybridLlamaSemanticLM])
def test_both_lm_implementations_expose_margin_objective(lm_class):
    torch.manual_seed(104)
    lm = lm_class(
        vocab_size=8,
        hidden_size=16,
        num_layers=1,
        num_attention_heads=4,
        dropout=0.0,
        max_position_embeddings=64,
    ).eval()
    prefix = torch.randn(1, 3, 16)
    targets = torch.tensor([[0, 0, 1, 2]])

    output = lm(
        prefix,
        targets,
        transition_predecessor_margin=1.5,
        transition_predecessor_margin_weight=0.25,
    )

    torch.testing.assert_close(
        output["objective_loss"],
        output["ce_objective_loss"]
        + 0.25 * output["transition_predecessor_margin_loss"],
    )
    assert torch.isfinite(output["transition_predecessor_rate"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("transition_predecessor_margin", True),
        ("transition_predecessor_margin", -1.0),
        ("transition_predecessor_margin", float("nan")),
        ("transition_predecessor_margin_weight", False),
        ("transition_predecessor_margin_weight", -1.0),
        ("transition_predecessor_margin_weight", float("inf")),
    ],
)
def test_margin_runtime_rejects_invalid_values(field, value):
    with pytest.raises(ValueError, match=field):
        _loss_metrics(
            torch.zeros(1, 2, 3),
            torch.tensor([[0, 1]]),
            torch.ones(1, 2, dtype=torch.bool),
            **{field: value},
        )


@pytest.mark.parametrize("value", [True, -0.1, 1.1, float("nan")])
def test_loss_metrics_reject_invalid_label_smoothing(value):
    with pytest.raises(ValueError, match="label_smoothing"):
        _loss_metrics(
            torch.zeros(1, 2, 3),
            torch.tensor([[0, 1]]),
            torch.ones(1, 2, dtype=torch.bool),
            label_smoothing=value,
        )
