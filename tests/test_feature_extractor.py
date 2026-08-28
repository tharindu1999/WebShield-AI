from unittest.mock import patch

import pytest

from feature_definitions import FEATURE_NAMES, ordered_feature_values
from feature_extractor import URLSafetyError, extract_html_features, validate_public_url


PUBLIC_DNS = [(2, 1, 6, "", ("93.184.216.34", 443))]


def test_valid_https_url():
    with patch("feature_extractor.socket.getaddrinfo", return_value=PUBLIC_DNS):
        assert validate_public_url("https://example.com/page") == "https://example.com/page"


@pytest.mark.parametrize("url", ["not a url", "ftp://example.com/file", "https://"])
def test_invalid_url(url):
    with pytest.raises(URLSafetyError):
        validate_public_url(url)


def test_localhost_rejected():
    with pytest.raises(URLSafetyError):
        validate_public_url("http://localhost:8501")


@pytest.mark.parametrize("address", ["127.0.0.1", "10.0.0.2", "192.168.1.20", "169.254.1.1", "::1"])
def test_private_or_local_ip_rejected(address):
    family = 10 if ":" in address else 2
    with patch("feature_extractor.socket.getaddrinfo", return_value=[(family, 1, 6, "", (address, 443))]):
        with pytest.raises(URLSafetyError):
            validate_public_url(f"https://[{address}]" if ":" in address else f"https://{address}")


def test_public_ip_handling():
    with patch("feature_extractor.socket.getaddrinfo", return_value=[(2, 1, 6, "", ("8.8.8.8", 443))]):
        assert validate_public_url("https://8.8.8.8") == "https://8.8.8.8"


def test_html_title_form_fields_and_links():
    html = """
    <html><head><title>Example</title><meta name="description" content="Demo">
    <link rel="icon" href="/favicon.ico"><link rel="stylesheet" href="/main.css"></head>
    <body><form action="https://collector.example.net/send">
      <input type="hidden"><input type="password"><button>Send</button>
    </form><img src="a.png"><iframe></iframe><script src="a.js"></script>
    <a href="/inside">Self</a><a href="">Empty</a><a href="https://outside.test">External</a>
    </body></html>
    """
    features = extract_html_features(html, "https://example.com")
    assert features["HasTitle"] == 1
    assert features["HasExternalFormSubmit"] == 1
    assert features["HasSubmitButton"] == 1
    assert features["HasHiddenFields"] == 1
    assert features["HasPasswordField"] == 1
    assert features["NoOfSelfRef"] == 1
    assert features["NoOfEmptyRef"] == 1
    assert features["NoOfExternalRef"] == 1


def test_feature_vector_ordering():
    values = {name: index for index, name in enumerate(reversed(FEATURE_NAMES))}
    vector = ordered_feature_values(values)
    assert vector == [float(values[name]) for name in FEATURE_NAMES]


def test_model_input_feature_count():
    import joblib
    model = joblib.load("models/web_risk_model.pkl")
    assert model.n_features_in_ == len(FEATURE_NAMES)

