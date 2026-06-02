"""
test_autocomplete_interactive.py
Keystroke-by-keystroke autocomplete tester with debounce.

Usage:
    pip install readchar httpx python-dotenv
    python test_autocomplete_interactive.py

Controls:
    Type          — builds prefix and fires debounced request
    Backspace     — deletes last character
    Enter         — fires immediate request then resets prefix
    Esc / Ctrl+C  — quit
"""

import httpx
import os
import threading
import time
from dotenv import load_dotenv

load_dotenv()

try:
    import readchar
except ImportError:
    print("This script requires the readchar library.")
    print("Install it with:  pip install readchar")
    raise SystemExit(1)

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_URL = "http://127.0.0.1:8000"
API_KEY         = os.environ.get("API_KEY", "")
DEBOUNCE_SECS   = 0.15      # seconds to wait after last keypress
MIN_PREFIX_LEN  = 3         # server also enforces this

# Build headers — show clearly what we're sending
if API_KEY:
    HEADERS = {"X-API-Key": API_KEY}
    _key_preview = f"{API_KEY[:4]}{'*' * max(0, len(API_KEY) - 4)}"
else:
    HEADERS = {}
    _key_preview = "(none)"


# ── Display helpers ───────────────────────────────────────────────────────────
CATEGORY_ICONS = {
    "snomed":  "🧬",
    "geo":     "📍",
    "phase":   "🔬",
    "unknown": "❓",
}

MATCH_COLORS = {
    "exact":    "\033[32m",   # green
    "synonym":  "\033[34m",   # blue
    "fuzzy":    "\033[35m",   # magenta
    "semantic": "\033[33m",   # yellow
    "prefix":   "\033[36m",   # cyan
}
RESET = "\033[0m"
DIM   = "\033[2m"
BOLD  = "\033[1m"
RED   = "\033[31m"
YELLOW = "\033[33m"


def _color_match(match_type: str, text: str) -> str:
    return f"{MATCH_COLORS.get(match_type, '')}{text}{RESET}"


def _print_banner() -> None:
    print(f"""
{BOLD}{'=' * 62}{RESET}
  Autocomplete Keystroke Simulator  (debounced {int(DEBOUNCE_SECS * 1000)}ms)
{'=' * 62}
  API   : {BASE_URL}
  Key   : {_key_preview}
  Auth  : {'✓ header will be sent' if API_KEY else f'{RED}✗ no key — expect 401 if auth required{RESET}'}
{'─' * 62}
  Backspace  delete char    Enter  submit now + reset
  Esc        quit           Ctrl+C quit
{'=' * 62}
""")


# ── Core request ──────────────────────────────────────────────────────────────
def fetch_suggestions(prefix: str, *, immediate: bool = False) -> None:
    """
    GET /v1/autocomplete and pretty-print results.
    `immediate` just affects the label shown (Enter vs debounce).
    """
    stripped = prefix.strip()
    if len(stripped) < MIN_PREFIX_LEN:
        return

    label = "enter" if immediate else "debounce"

    try:
        t0 = time.perf_counter()
        response = httpx.get(
            f"{BASE_URL}/v1/autocomplete",
            params={"q": stripped, "limit": 7},
            headers=HEADERS,
            timeout=5.0,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # ── Handle non-2xx explicitly so the error message is actionable ──────
        if response.status_code == 401:
            print(
                f"\n  {RED}401 Unauthorized{RESET} — "
                f"API_KEY in your .env doesn't match the server.\n"
                f"  Sent:     X-API-Key: {_key_preview}\n"
                f"  Fix:      make sure .env API_KEY matches the server's configured key.\n"
            )
            print(f"Query: {prefix}", end="", flush=True)
            return

        if response.status_code == 403:
            print(
                f"\n  {RED}403 Forbidden{RESET} — "
                f"key recognised but access denied.\n"
            )
            print(f"Query: {prefix}", end="", flush=True)
            return

        if response.status_code == 503:
            print(
                f"\n  {YELLOW}503 Service Unavailable{RESET} — "
                f"AutocompleteOrchestrator did not initialise (check uvicorn logs).\n"
            )
            print(f"Query: {prefix}", end="", flush=True)
            return

        if not response.is_success:
            print(
                f"\n  {RED}HTTP {response.status_code}{RESET} — {response.text[:200]}\n"
            )
            print(f"Query: {prefix}", end="", flush=True)
            return

        # ── Parse payload ─────────────────────────────────────────────────────
        data        = response.json()
        suggestions = data.get("suggestions", [])
        tier        = data.get("tier_used", "unknown")
        srv_ms      = data.get("latency_ms")          # server-side if exposed
        latency_str = f"{elapsed_ms:.0f}ms (client)"
        if srv_ms is not None:
            latency_str += f" / {srv_ms}ms (server)"

        print(f"\n  [{label}] {latency_str}  tier={tier}  {len(suggestions)} result(s)")

        if not suggestions:
            print(f"  {DIM}(no suggestions for '{stripped}'){RESET}")
        else:
            for i, s in enumerate(suggestions, 1):
                icon     = CATEGORY_ICONS.get(s.get("category", ""), "  ")
                mtype    = s.get("match_type", "")
                conf     = s.get("confidence", 0.0)
                display  = s.get("display", "")
                completion = s.get("completion", "")

                print(
                    f"  {i}. {icon} {_color_match(mtype, display)}"
                    f"  {DIM}[{s.get('category','')}]"
                    f" [{mtype}]"
                    f" {conf:.2f}{RESET}"
                )
                if completion and completion != display:
                    print(f"       {DIM}→ {completion}{RESET}")

        print()
        print(f"Query: {prefix}", end="", flush=True)

    except httpx.ConnectError:
        print(
            f"\n  {RED}Connection refused{RESET} — "
            f"is uvicorn running on {BASE_URL}?\n"
        )
        print(f"Query: {prefix}", end="", flush=True)

    except httpx.TimeoutException:
        print(
            f"\n  {YELLOW}Timeout{RESET} — "
            f"server took >5 s. Check uvicorn logs.\n"
        )
        print(f"Query: {prefix}", end="", flush=True)

    except Exception as exc:
        print(
            f"\n  {RED}Unexpected error{RESET} "
            f"{type(exc).__name__}: {exc}\n"
        )
        print(f"Query: {prefix}", end="", flush=True)


# ── Debounce helper ───────────────────────────────────────────────────────────
class _Debouncer:
    """Cancel-and-reschedule timer so only the last keypress fires a request."""

    def __init__(self, delay: float) -> None:
        self._delay  = delay
        self._timer: threading.Timer | None = None
        self._lock   = threading.Lock()

    def schedule(self, prefix: str) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            if len(prefix.strip()) >= MIN_PREFIX_LEN:
                self._timer = threading.Timer(
                    self._delay, fetch_suggestions, args=[prefix]
                )
                self._timer.daemon = True
                self._timer.start()

    def cancel(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()


# ── Main loop ─────────────────────────────────────────────────────────────────
def main() -> None:
    _print_banner()

    # Warn early if no key is configured — saves confusion later
    if not API_KEY:
        print(
            f"  {YELLOW}Warning:{RESET} API_KEY is not set in your .env file.\n"
            f"  If the server requires auth you will get 401 on every request.\n"
            f"  Add  API_KEY=<your_key>  to .env and rerun this script.\n"
        )

    current_prefix = ""
    debouncer      = _Debouncer(DEBOUNCE_SECS)

    print("Query: ", end="", flush=True)

    try:
        while True:
            key = readchar.readkey()

            # ── Quit ──────────────────────────────────────────────────────────
            if key in (readchar.key.CTRL_C, readchar.key.ESC):
                print("\nGoodbye.")
                break

            # ── Backspace / Delete ────────────────────────────────────────────
            if key in (readchar.key.BACKSPACE, readchar.key.DELETE):
                if current_prefix:
                    current_prefix = current_prefix[:-1]
                    # Overwrite the line cleanly
                    print(
                        f"\rQuery: {current_prefix} "
                        f"\rQuery: {current_prefix}",
                        end="",
                        flush=True,
                    )
                    debouncer.schedule(current_prefix)
                continue

            # ── Enter — fire immediately, then reset ──────────────────────────
            if key == readchar.key.ENTER:
                stripped = current_prefix.strip().lower()
                if stripped in ("quit", "exit"):
                    print("\nGoodbye.")
                    break
                debouncer.cancel()
                print()                          # newline before results
                fetch_suggestions(current_prefix, immediate=True)
                current_prefix = ""
                print("Query: ", end="", flush=True)
                continue

            # ── Printable character ───────────────────────────────────────────
            if len(key) == 1 and key.isprintable():
                current_prefix += key
                print(f"\rQuery: {current_prefix}", end="", flush=True)
                debouncer.schedule(current_prefix)

    except KeyboardInterrupt:
        print("\nGoodbye.")
    finally:
        debouncer.cancel()


if __name__ == "__main__":
    main()