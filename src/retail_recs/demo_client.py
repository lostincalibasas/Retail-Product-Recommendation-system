"""HTTP-only frontend access to the existing recommendation service."""
import json
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from pydantic import ValidationError

from .api_models import Recommendations


class DemoError(Exception):
    """An actionable, user-facing service error."""


def fetch_recommendations(base_url: str, household_id: int) -> dict:
    try:
        with urlopen(f"{base_url.rstrip('/')}/recommendations/{household_id}", timeout=15) as response:
            payload = json.load(response)
    except HTTPError as exc:
        if exc.code == 404:
            raise DemoError("This household is not available in the historical snapshot. Try another household ID.") from exc
        raise DemoError("The recommendation service could not complete the request. Please try again.") from exc
    except (URLError, TimeoutError, ConnectionError) as exc:
        raise DemoError("Cannot reach the recommendation service. Start FastAPI and try again.") from exc
    except (ValueError, UnicodeError) as exc:
        raise DemoError("The recommendation service returned an unreadable response. Please try again.") from exc
    try:
        result = Recommendations.model_validate(payload)
        if (result.household_id != household_id or len(result.recommendations) != 5
                or len({p.product_id for p in result.recommendations}) != 5):
            raise ValueError("Unexpected household or recommendation count")
    except (ValidationError, ValueError) as exc:
        raise DemoError("The recommendation service returned an unexpected response. Please try again.") from exc
    return result.model_dump()
