"""Safe passive webpage retrieval and WebShield feature extraction."""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

import requests
from bs4 import BeautifulSoup

from feature_definitions import FEATURE_NAMES, extract_url_features, ordered_feature_values


MAX_REDIRECTS = 5
MAX_BODY_BYTES = 2 * 1024 * 1024
CONNECT_TIMEOUT_SECONDS = 5
READ_TIMEOUT_SECONDS = 10
USER_AGENT = "WebShield-AI/1.0 (passive university research classifier)"


class URLSafetyError(ValueError):
    """Raised when a URL violates validation or public-network requirements."""


class WebsiteFetchError(RuntimeError):
    """Raised when a validated website cannot be safely retrieved as HTML."""


@dataclass(frozen=True)
class ExtractionResult:
    requested_url: str
    final_url: str
    status_code: int
    features: dict[str, int]

    @property
    def vector(self) -> list[float]:
        return ordered_feature_values(self.features)


def _is_public_ip(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    return ip.is_global


def validate_public_url(url: str) -> str:
    """Validate syntax and confirm every resolved address is globally routable."""
    if not isinstance(url, str) or not url.strip():
        raise URLSafetyError("Enter a complete HTTP or HTTPS URL.")
    url = url.strip()
    parsed = urlsplit(url)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise URLSafetyError("Only HTTP and HTTPS URLs are accepted.")
    if not parsed.hostname or parsed.username or parsed.password:
        raise URLSafetyError("The URL must contain a hostname and no embedded credentials.")
    hostname = parsed.hostname.rstrip(".").lower()
    if hostname == "localhost" or hostname.endswith(".localhost") or "." not in hostname and not _looks_like_ip(hostname):
        raise URLSafetyError("Local hostnames are not allowed.")
    try:
        port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    except ValueError as exc:
        raise URLSafetyError("The URL contains an invalid port.") from exc
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)}
    except socket.gaierror as exc:
        raise URLSafetyError("The hostname could not be resolved.") from exc
    if not addresses or any(not _is_public_ip(address) for address in addresses):
        raise URLSafetyError("The hostname must resolve only to public internet addresses.")
    return url


def _looks_like_ip(hostname: str) -> bool:
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        return False


def _same_host(reference: str, base_url: str) -> bool:
    resolved = urlsplit(urljoin(base_url, reference))
    return (resolved.hostname or "").lower() == (urlsplit(base_url).hostname or "").lower()


def extract_html_features(html: str, page_url: str) -> dict[str, int]:
    """Extract straightforward DOM presence/count features without executing JS."""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.find("title")
    favicon = soup.find("link", rel=lambda value: value and "icon" in [str(x).lower() for x in (value if isinstance(value, list) else [value])])
    description = soup.find("meta", attrs={"name": lambda value: value and str(value).lower() == "description"})
    forms = soup.find_all("form")
    form_actions = [str(form.get("action", "")).strip() for form in forms]
    external_forms = sum(bool(action) and not action.startswith(("#", "javascript:")) and not _same_host(action, page_url) for action in form_actions)
    submit_controls = soup.select('input[type="submit"], button[type="submit"], button:not([type])')
    anchors = soup.find_all("a")
    self_refs = empty_refs = external_refs = 0
    for anchor in anchors:
        href = str(anchor.get("href", "")).strip()
        if not href or href == "#" or href.lower().startswith("javascript:"):
            empty_refs += 1
        elif _same_host(href, page_url):
            self_refs += 1
        else:
            external_refs += 1

    return {
        "HasTitle": int(bool(title and title.get_text(strip=True))),
        "HasFavicon": int(favicon is not None),
        "HasDescription": int(description is not None and bool(description.get("content", "").strip())),
        "NoOfiFrame": len(soup.find_all("iframe")),
        "HasExternalFormSubmit": int(external_forms > 0),
        "HasSubmitButton": int(bool(submit_controls)),
        "HasHiddenFields": int(bool(soup.select('input[type="hidden"]'))),
        "HasPasswordField": int(bool(soup.select('input[type="password"]'))),
        "NoOfImage": len(soup.find_all("img")),
        "NoOfCSS": len(soup.find_all("link", rel=lambda value: value and "stylesheet" in [str(x).lower() for x in (value if isinstance(value, list) else [value])])),
        "NoOfJS": len(soup.find_all("script")),
        "NoOfSelfRef": self_refs,
        "NoOfEmptyRef": empty_refs,
        "NoOfExternalRef": external_refs,
    }


def fetch_and_extract(url: str, session: requests.Session | None = None) -> ExtractionResult:
    """Validate every hop, stream a bounded HTML response, and extract features."""
    current_url = validate_public_url(url)
    client = session or requests.Session()
    client.trust_env = False
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}

    for redirect_count in range(MAX_REDIRECTS + 1):
        try:
            response = client.get(
                current_url,
                headers=headers,
                timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
                allow_redirects=False,
                stream=True,
                verify=True,
            )
        except requests.exceptions.SSLError as exc:
            raise WebsiteFetchError("TLS certificate validation failed.") from exc
        except requests.exceptions.Timeout as exc:
            raise WebsiteFetchError("The website request timed out.") from exc
        except requests.exceptions.RequestException as exc:
            raise WebsiteFetchError(f"The website could not be retrieved: {exc}") from exc

        if response.is_redirect or response.is_permanent_redirect:
            response.close()
            if redirect_count >= MAX_REDIRECTS:
                raise WebsiteFetchError(f"The website exceeded the {MAX_REDIRECTS}-redirect limit.")
            location = response.headers.get("Location")
            if not location:
                raise WebsiteFetchError("The website returned a redirect without a destination.")
            current_url = validate_public_url(urljoin(current_url, location))
            continue

        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            response.close()
            raise WebsiteFetchError(f"The website returned HTTP {response.status_code}.") from exc

        content_type = response.headers.get("Content-Type", "").lower()
        if not (content_type.startswith("text/html") or "application/xhtml+xml" in content_type):
            response.close()
            raise WebsiteFetchError("The response is not HTML compatible.")

        body = bytearray()
        try:
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if chunk:
                    body.extend(chunk)
                    if len(body) > MAX_BODY_BYTES:
                        raise WebsiteFetchError(f"The HTML response exceeds the {MAX_BODY_BYTES // (1024 * 1024)} MB limit.")
        finally:
            response.close()
        encoding = response.encoding or "utf-8"
        html = bytes(body).decode(encoding, errors="replace")
        features = extract_url_features(current_url)
        features.update(extract_html_features(html, current_url))
        if list(features) != FEATURE_NAMES:
            raise RuntimeError("Feature extractor order does not match the canonical model schema.")
        return ExtractionResult(url, current_url, response.status_code, features)

    raise WebsiteFetchError("Unable to retrieve the website.")

