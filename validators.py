"""Shared field validation and record analysis (pure functions, no I/O).

Two entry points with different strictness:
  validate_field(name, value) - for values the CANDIDATE supplies. Conservative
      normalisation ("b tech" -> "B.Tech", "8.4" -> 8.4). Returns the value to store.
  check_stored(name, value)   - for values ALREADY in the database. A value is valid
      only if it is already in canonical form (right type, right spelling).

Reasons never contain the value that was checked, so they are safe to print.
"""
import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Any
from urllib.parse import urlparse

# --- Field catalogue (from the real /v1/schema and the assignment) -------------
MANDATORY_FIELDS = (
    "full_name", "phone", "email", "college", "degree", "branch", "graduation_year",
    "cgpa", "preferred_location", "available_from", "internship_duration_months",
)
OPTIONAL_FIELDS = ("linkedin_url", "github_url", "skills")
TRACKING_FIELDS = ("collection_status", "attempt_count", "last_contacted_at", "notes")
READ_ONLY_FIELDS = frozenset({"id", "phone", "version", "updated_at"})
WRITABLE_FIELDS = frozenset(
    (set(MANDATORY_FIELDS) | set(OPTIONAL_FIELDS) | set(TRACKING_FIELDS)) - READ_ONLY_FIELDS
)

DEGREES = ("B.Tech", "B.E.", "B.Sc", "BCA", "MBA Tech", "MCA", "M.Tech", "Other")
LOCATIONS = ("Mumbai", "Bengaluru", "Remote", "No preference")
DURATIONS = (2, 3, 6)
GRADUATION_YEARS = (2026, 2028)
DATE_MIN, DATE_MAX = date(2026, 1, 1), date(2027, 12, 31)
COLLECTION_STATUSES = ("pending", "in_progress", "completed", "no_answer", "refused", "needs_human")
MAX_SKILLS, MAX_SKILL_LEN, MAX_NOTES_LEN = 30, 50, 4000


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    normalized_value: Any
    reason: str | None


def _ok(value) -> ValidationResult:
    return ValidationResult(True, value, None)


def _bad(reason: str) -> ValidationResult:
    return ValidationResult(False, None, reason)


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _to_int(v) -> int | None:
    """Whole number from an int, an integral float, or a plain digit string."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v) if math.isfinite(v) and v.is_integer() else None
    if isinstance(v, str) and re.fullmatch(r"[+-]?\d+", v.strip()):
        return int(v.strip())
    return None


def is_empty(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip()) or value == []


# --- Individual validators (candidate input -> ValidationResult) ---------------
def _text(label: str, lo: int, hi: int):
    def check(v):
        if not isinstance(v, str):
            return _bad(f"{label} must be text")
        s = v.strip()
        if not lo <= len(s) <= hi:
            return _bad(f"{label} must be {lo}-{hi} characters")
        if not any(c.isalnum() for c in s):
            return _bad(f"{label} must contain letters or digits")
        return _ok(s)
    return check


_EMAIL_RE = re.compile(
    r"^[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*"
    r"@(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,}$"
)


def _email(v):
    if not isinstance(v, str):
        return _bad("email must be text")
    s = v.strip()
    if len(s) > 254:
        return _bad("email is longer than 254 characters")
    if not _EMAIL_RE.match(s) or len(s.split("@")[0]) > 64:
        return _bad("not a valid email address")
    return _ok(s)


def _squash(s: str, keep: str = "a-z") -> str:
    return re.sub(rf"[^{keep}]", "", s.lower())


_DEGREE_ALIASES = {
    "btech": "B.Tech", "bacheloroftechnology": "B.Tech",
    "be": "B.E.", "bachelorofengineering": "B.E.",
    "bsc": "B.Sc", "bachelorofscience": "B.Sc",
    "bca": "BCA", "bachelorofcomputerapplications": "BCA",
    "mbatech": "MBA Tech",
    "mca": "MCA", "masterofcomputerapplications": "MCA",
    "mtech": "M.Tech", "masteroftechnology": "M.Tech",
    "other": "Other",
}


def _degree(v):
    if not isinstance(v, str):
        return _bad("degree must be text")
    match = _DEGREE_ALIASES.get(_squash(v))
    if match:
        return _ok(match)
    return _bad(f"degree must be one of: {', '.join(DEGREES)}")


_LOCATION_ALIASES = {
    "mumbai": "Mumbai", "bombay": "Mumbai",
    "bengaluru": "Bengaluru", "bangalore": "Bengaluru", "bengalooru": "Bengaluru",
    "remote": "Remote", "wfh": "Remote", "workfromhome": "Remote",
    "nopreference": "No preference", "nopref": "No preference", "any": "No preference",
    "anywhere": "No preference", "anylocation": "No preference",
    "dontmind": "No preference", "dontcare": "No preference", "doesntmatter": "No preference",
    "dontmatter": "No preference",
}


def _location(v):
    if not isinstance(v, str):
        return _bad("preferred_location must be text")
    match = _LOCATION_ALIASES.get(_squash(v))
    if match:
        return _ok(match)
    return _bad(f"preferred_location must be one of: {', '.join(LOCATIONS)}")


def _graduation_year(v):
    year = _to_int(v)
    if year is None:
        return _bad("graduation_year must be a full 4-digit year (a year is never guessed)")
    lo, hi = GRADUATION_YEARS
    if not lo <= year <= hi:
        return _bad(f"graduation_year must be between {lo} and {hi}")
    return _ok(year)


def _cgpa(v):
    if isinstance(v, bool):
        return _bad("cgpa must be a number")
    if _is_num(v):
        if not math.isfinite(v):
            return _bad("cgpa must be a finite number")
        dec = Decimal(str(v))
    elif isinstance(v, str):
        s = v.strip().lower()
        if "%" in s or "percent" in s:
            return _bad("cgpa must be on the 10-point scale, not a percentage")
        m = re.fullmatch(r"(\d+(?:\.\d+)?|\.\d+)\s*(?:/\s*(\d+(?:\.\d+)?)|out of\s*(\d+(?:\.\d+)?))?", s)
        if not m:
            return _bad("cgpa must be a number such as 8.4")
        denominator = m.group(2) or m.group(3)
        if denominator is not None and Decimal(denominator) != 10:
            return _bad("cgpa must be on the 10-point scale")
        dec = Decimal(m.group(1))
    else:
        return _bad("cgpa must be a number")
    if not Decimal(0) <= dec <= Decimal(10):
        return _bad("cgpa must be between 0.0 and 10.0")
    rounded = dec.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    if rounded > 10:
        return _bad("cgpa must be between 0.0 and 10.0")
    return _ok(float(rounded))


_MONTHS = {}
for _n, _names in enumerate(
    [("january", "jan"), ("february", "feb"), ("march", "mar"), ("april", "apr"), ("may",),
     ("june", "jun"), ("july", "jul"), ("august", "aug"), ("september", "sep", "sept"),
     ("october", "oct"), ("november", "nov"), ("december", "dec")], start=1):
    for _name in _names:
        _MONTHS[_name] = _n

_DATE_HELP = "give an exact date, e.g. 14 July 2026"


def _build_date(year: int, month: int, day: int):
    try:
        d = date(year, month, day)
    except ValueError:
        return _bad("not a real calendar date")
    if not DATE_MIN <= d <= DATE_MAX:
        return _bad(f"available_from must be between {DATE_MIN} and {DATE_MAX}")
    return _ok(d.isoformat())


def _available_from(v):
    if not isinstance(v, str):
        return _bad("available_from must be a date written as text")
    s = re.sub(r"(?<=\d)(?:st|nd|rd|th)\b", "", v.strip().lower())
    s = re.sub(r"\s+", " ", s.replace(",", " ")).strip()

    m = re.fullmatch(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", s)  # year first: unambiguous
    if m:
        return _build_date(*map(int, m.groups()))

    m = re.fullmatch(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})", s)  # 01/02/2027 style
    if m:
        a, b, year = map(int, m.groups())
        if a <= 12 and b <= 12 and a != b:
            return _bad("date is ambiguous (day/month order unclear); " + _DATE_HELP)
        day, month = (a, b) if a > 12 else (b, a) if b > 12 else (a, b)
        return _build_date(year, month, day)

    m = re.fullmatch(r"(\d{1,2}) ?-?/?([a-z]+) ?-?/?(\d{4})", s)  # 1 march 2027
    if m and m.group(2) in _MONTHS:
        return _build_date(int(m.group(3)), _MONTHS[m.group(2)], int(m.group(1)))

    m = re.fullmatch(r"([a-z]+) ?-?/?(\d{1,2})[ \-/](\d{4})", s)  # march 1 2027
    if m and m.group(1) in _MONTHS:
        return _build_date(int(m.group(3)), _MONTHS[m.group(1)], int(m.group(2)))

    if re.fullmatch(r"[a-z]+ \d{4}|\d{4} [a-z]+", s):
        return _bad("the day is missing; " + _DATE_HELP)
    if re.fullmatch(r"\d{1,2} [a-z]+|[a-z]+ \d{1,2}", s):
        return _bad("the year is missing; " + _DATE_HELP)
    return _bad("could not read this as an exact date; " + _DATE_HELP)


_WORD_NUMBERS = {"two": 2, "three": 3, "six": 6}


def _duration(v):
    months = None
    if isinstance(v, str):
        m = re.fullmatch(r"(\d+|two|three|six)(?:\s*(?:months?|mos?))?", v.strip().lower())
        if m:
            months = _WORD_NUMBERS.get(m.group(1)) or int(m.group(1))
    else:
        months = _to_int(v)
    if months in DURATIONS:
        return _ok(months)
    return _bad("internship_duration_months must be 2, 3 or 6")


def _parse_url(v, label: str):
    """Return (url_with_scheme, hostname, path) or a ValidationResult error."""
    if not isinstance(v, str):
        return _bad(f"{label} must be text")
    s = v.strip()
    if not s or re.search(r"\s", s):
        return _bad(f"{label} must be a single URL without spaces")
    if not re.match(r"https?://", s, re.I):
        if "://" in s:
            return _bad(f"{label} must start with http:// or https://")
        s = "https://" + s
    try:
        parsed = urlparse(s)
        host = parsed.hostname
    except ValueError:
        return _bad(f"{label} is not a valid URL")
    if not host or parsed.username or parsed.password:
        return _bad(f"{label} is not a valid URL")
    return s, host.lower(), parsed.path


def _linkedin(v):
    parsed = _parse_url(v, "linkedin_url")
    if isinstance(parsed, ValidationResult):
        return parsed
    url, host, path = parsed
    if not (host == "linkedin.com" or host.endswith(".linkedin.com")):
        return _bad("linkedin_url must be a linkedin.com URL")
    if not path.strip("/"):
        return _bad("linkedin_url must point to a profile, not the home page")
    return _ok(url)


def _github(v):
    parsed = _parse_url(v, "github_url")
    if isinstance(parsed, ValidationResult):
        return parsed
    url, host, path = parsed
    if host not in ("github.com", "www.github.com"):
        return _bad("github_url must be a github.com URL")
    segments = [p for p in path.split("/") if p]
    if len(segments) != 1 or not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})", segments[0]):
        return _bad("github_url must be a profile URL like github.com/username")
    return _ok(url)


def _skills(v):
    if isinstance(v, str):
        items = [p.strip() for p in v.split(",") if p.strip()]
    elif isinstance(v, (list, tuple)):
        items = list(v)
    else:
        return _bad("skills must be a list of strings")
    if not items:
        return _bad("skills is empty")
    if len(items) > MAX_SKILLS:
        return _bad(f"skills can have at most {MAX_SKILLS} items")
    cleaned = []
    for item in items:
        if not isinstance(item, str):
            return _bad("every skill must be text")
        item = item.strip()
        if not 1 <= len(item) <= MAX_SKILL_LEN:
            return _bad(f"each skill must be 1-{MAX_SKILL_LEN} characters")
        cleaned.append(item)
    return _ok(cleaned)


# Tracking fields (written by the agent, not collected from the candidate)
def _status(v):
    s = re.sub(r"[\s-]+", "_", v.strip().lower()) if isinstance(v, str) else None
    if s in COLLECTION_STATUSES:
        return _ok(s)
    return _bad(f"collection_status must be one of: {', '.join(COLLECTION_STATUSES)}")


def _attempt_count(v):
    n = _to_int(v)
    if n is not None and 0 <= n <= 50:
        return _ok(n)
    return _bad("attempt_count must be an integer 0-50")


def _last_contacted(v):
    if isinstance(v, str):
        try:
            dt = datetime.fromisoformat(v.strip())
        except ValueError:
            dt = None
        if dt is not None and dt.tzinfo is not None:
            return _ok(dt.isoformat())
    return _bad("last_contacted_at must be ISO 8601 with a timezone offset")


def _notes(v):
    if isinstance(v, str) and 1 <= len(v) <= MAX_NOTES_LEN:
        return _ok(v)
    return _bad(f"notes must be 1-{MAX_NOTES_LEN} characters")


VALIDATORS = {
    "full_name": _text("full_name", 2, 100),
    "email": _email,
    "college": _text("college", 2, 150),
    "degree": _degree,
    "branch": _text("branch", 2, 100),
    "graduation_year": _graduation_year,
    "cgpa": _cgpa,
    "preferred_location": _location,
    "available_from": _available_from,
    "internship_duration_months": _duration,
    "linkedin_url": _linkedin,
    "github_url": _github,
    "skills": _skills,
    "collection_status": _status,
    "attempt_count": _attempt_count,
    "last_contacted_at": _last_contacted,
    "notes": _notes,
}

E164_RE = re.compile(r"^\+[1-9]\d{6,14}$")


# --- Public API -----------------------------------------------------------------
def validate_field(name: str, value: Any) -> ValidationResult:
    """Validate and normalise a value that is about to be WRITTEN."""
    if name in READ_ONLY_FIELDS:
        return _bad(f"'{name}' is read-only and can never be written")
    validator = VALIDATORS.get(name)
    if validator is None:
        return _bad(f"'{name}' is not a known writable field")
    if value is None:
        return _bad("no value given")
    return validator(value)


def _same(stored, normalized) -> bool:
    if _is_num(stored) and _is_num(normalized):
        return stored == normalized
    return type(stored) is type(normalized) and stored == normalized


def check_stored(name: str, value: Any) -> ValidationResult:
    """Strictly validate a value already stored in the database."""
    if name == "phone":
        if isinstance(value, str) and E164_RE.match(value):
            return _ok(value)
        return _bad("phone is not in E.164 format")
    result = validate_field(name, value)
    if result.valid and not _same(value, result.normalized_value):
        return _bad("stored value is not in canonical form (wrong type or spelling)")
    return result


@dataclass
class RecordAnalysis:
    missing: list[str]                  # mandatory, empty
    invalid: dict[str, str]             # mandatory, populated but invalid -> reason
    optional_populated: list[str]
    optional_invalid: dict[str, str]    # never blocks completion
    collection_status: Any = None
    attempt_count: Any = None

    @property
    def blocked_read_only(self) -> list[str]:
        """Problem fields the agent can't fix because they are read-only (e.g. phone)."""
        return [f for f in [*self.missing, *self.invalid] if f in READ_ONLY_FIELDS]

    @property
    def fields_to_collect(self) -> list[str]:
        """Mandatory fields to collect or correct, in schema order, writable only."""
        problems = set(self.missing) | set(self.invalid)
        return [f for f in MANDATORY_FIELDS if f in problems and f not in READ_ONLY_FIELDS]

    @property
    def needs_collection(self) -> bool:
        return bool(self.fields_to_collect)


def analyse_record(record: dict, mandatory_fields=MANDATORY_FIELDS) -> RecordAnalysis:
    missing, invalid = [], {}
    for name in mandatory_fields:
        value = record.get(name)
        if is_empty(value):
            missing.append(name)
        else:
            result = check_stored(name, value)
            if not result.valid:
                invalid[name] = result.reason
    optional_populated, optional_invalid = [], {}
    for name in OPTIONAL_FIELDS:
        value = record.get(name)
        if not is_empty(value):
            optional_populated.append(name)
            result = check_stored(name, value)
            if not result.valid:
                optional_invalid[name] = result.reason
    return RecordAnalysis(
        missing=missing,
        invalid=invalid,
        optional_populated=optional_populated,
        optional_invalid=optional_invalid,
        collection_status=record.get("collection_status"),
        attempt_count=record.get("attempt_count"),
    )
