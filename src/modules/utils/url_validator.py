# src/modules/utils/url_validator.py

import re
from typing import Optional, Dict, Any


def validate_url(url: str) -> bool:
    """Validiert URLs mit regulärem Ausdruck nach RFC 1738"""
    regex = re.compile(
        r'^(?:http|https)://'  # http:// oder https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # Domain...
        r'localhost'  # ...oder localhost
        r'(?::\d+)?'  # Optionaler Port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    return bool(regex.match(url))