import pytest

from train import configured_seed


def test_configured_seed_defaults_and_accepts_explicit_integer():
    assert configured_seed({}) == 3407
    assert configured_seed({"seed": 7}) == 7


@pytest.mark.parametrize("value", [True, 3.5, "3407"])
def test_configured_seed_rejects_non_integer_values(value):
    with pytest.raises(ValueError, match="seed must be an integer"):
        configured_seed({"seed": value})
