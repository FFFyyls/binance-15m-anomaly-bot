import pytest

from scripts.manual import submit_test_stop_order


def test_testnet_order_does_not_require_live_confirmation() -> None:
    assert hasattr(
        submit_test_stop_order,
        "validate_order_environment",
    ), "production-order safety guard is missing"

    submit_test_stop_order.validate_order_environment(
        use_testnet=True,
        confirm_live_order=False,
    )


def test_production_order_is_rejected_without_explicit_confirmation() -> None:
    assert hasattr(
        submit_test_stop_order,
        "validate_order_environment",
    ), "production-order safety guard is missing"

    with pytest.raises(RuntimeError, match="production order"):
        submit_test_stop_order.validate_order_environment(
            use_testnet=False,
            confirm_live_order=False,
        )


def test_production_order_accepts_explicit_confirmation() -> None:
    assert hasattr(
        submit_test_stop_order,
        "validate_order_environment",
    ), "production-order safety guard is missing"

    submit_test_stop_order.validate_order_environment(
        use_testnet=False,
        confirm_live_order=True,
    )
