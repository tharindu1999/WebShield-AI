"""Canonical WebShield model feature schema and deterministic URL features."""

from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit


URL_FEATURES = [
    "URLLength",
    "DomainLength",
    "IsDomainIP",
    "TLDLength",
    "NoOfSubDomain",
    "IsHTTPS",
]

HTML_FEATURES = [
    "HasTitle",
    "HasFavicon",
    "HasDescription",
    "NoOfiFrame",
    "HasExternalFormSubmit",
    "HasSubmitButton",
    "HasHiddenFields",
    "HasPasswordField",
    "NoOfImage",
    "NoOfCSS",
    "NoOfJS",
    "NoOfSelfRef",
    "NoOfEmptyRef",
    "NoOfExternalRef",
]

FEATURE_NAMES = URL_FEATURES + HTML_FEATURES


def extract_url_features(url: str) -> dict[str, int]:
    """Return deterministic URL features used identically in training and inference."""
    parsed = urlsplit(url)
    hostname = (parsed.hostname or "").rstrip(".").lower()
    try:
        ipaddress.ip_address(hostname)
        is_ip = 1
    except ValueError:
        is_ip = 0

    labels = [label for label in hostname.split(".") if label]
    tld_length = 0 if is_ip or len(labels) < 2 else len(labels[-1])
    subdomains = 0 if is_ip else max(0, len(labels) - 2)
    return {
        "URLLength": len(url),
        "DomainLength": len(hostname),
        "IsDomainIP": is_ip,
        "TLDLength": tld_length,
        "NoOfSubDomain": subdomains,
        "IsHTTPS": int(parsed.scheme.lower() == "https"),
    }


def ordered_feature_values(features: dict[str, int | float]) -> list[float]:
    """Create a model vector and fail loudly if any canonical feature is absent."""
    missing = [name for name in FEATURE_NAMES if name not in features]
    if missing:
        raise ValueError(f"Missing model features: {', '.join(missing)}")
    return [float(features[name]) for name in FEATURE_NAMES]

