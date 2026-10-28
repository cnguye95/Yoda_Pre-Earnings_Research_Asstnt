"""One shared web session that quietly retries a failed request once.

Servers occasionally answer with a temporary error (HTTP 500, 502, 503 or
504) that succeeds a moment later. Rather than every module writing its own
"try again" loop, they all send GET requests through `session`, which uses
urllib3's tested Retry logic:

  - A server error (5xx) or a dropped connection is retried once, straight
    away. urllib3 never waits before a first retry, except when a 503 reply
    includes a Retry-After header saying how long to wait.
  - A client error (4xx — bad key, blocked page, not found) is not retried,
    because asking again would get the same answer.
  - If the retry also fails, the last error response is handed back as
    normal, so callers still see the real status code and can report it.
"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# The retry rules described above. allowed_methods limits retries to GET,
# which is safe to repeat because it never changes anything on the server.
RETRY_POLICY = Retry(
    total=1,
    status_forcelist=(500, 502, 503, 504),
    allowed_methods=frozenset({"GET"}),
    raise_on_status=False,
)


def make_session() -> requests.Session:
    """Build a requests.Session that applies RETRY_POLICY to every web address."""
    new_session = requests.Session()
    adapter = HTTPAdapter(max_retries=RETRY_POLICY)
    new_session.mount("https://", adapter)
    new_session.mount("http://", adapter)
    return new_session


# The single session the rest of Yoda imports. Sharing it also reuses open
# connections, which makes repeated calls to the same site faster.
session = make_session()
