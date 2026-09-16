from src.data_loader import normalize_gateway_id


def test_gateway_id_normalisation_accepts_both_formats():
    assert normalize_gateway_id('06:39:EA:56:02:C1') == '0639EA5602C1'
    assert normalize_gateway_id('0639ea5602c1') == '0639EA5602C1'
