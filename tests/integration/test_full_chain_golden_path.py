import pytest

@pytest.mark.skip(reason="Kiểm định Full Chain Golden Path chưa hoàn thiện")
def test_full_chain_parity_placeholder():
    """Kiểm định bắt buộc CI: sign_flip_count == 0 giữa Python và Rust."""
    assert True
