import pytest

from validators import (
    READ_ONLY_FIELDS, WRITABLE_FIELDS, analyse_record, check_stored, validate_field,
)


def ok(name, value, expected):
    r = validate_field(name, value)
    assert r.valid, (name, r.reason)
    assert r.normalized_value == expected
    assert type(r.normalized_value) is type(expected)
    assert r.reason is None


def bad(name, value):
    r = validate_field(name, value)
    assert not r.valid and r.normalized_value is None and r.reason
    assert len(str(value)) < 4 or str(value) not in r.reason  # reasons don't echo values


# --- read-only / unknown ---
@pytest.mark.parametrize("name", sorted(READ_ONLY_FIELDS))
def test_read_only_fields_rejected(name):
    r = validate_field(name, "+919999999999")
    assert not r.valid and "read-only" in r.reason


def test_phone_never_writable():
    assert "phone" not in WRITABLE_FIELDS
    assert {"id", "version", "updated_at"}.isdisjoint(WRITABLE_FIELDS)


def test_unknown_field_and_none_rejected():
    assert not validate_field("shoe_size", 9).valid
    assert not validate_field("email", None).valid


# --- strings ---
def test_full_name_bounds():
    ok("full_name", "  Aarav Shah ", "Aarav Shah")
    ok("full_name", "Al", "Al")
    bad("full_name", "A")
    bad("full_name", "x" * 101)
    ok("full_name", "x" * 100, "x" * 100)
    bad("full_name", 123)
    bad("full_name", "  . ")


def test_college_and_branch_bounds():
    ok("college", "IIT", "IIT")
    bad("college", "I")
    ok("college", "c" * 150, "c" * 150)
    bad("college", "c" * 151)
    ok("branch", "Computer Engineering", "Computer Engineering")
    bad("branch", "C")
    bad("branch", "b" * 101)


# --- email ---
@pytest.mark.parametrize("value", ["a@b.co", "first.last+tag@sub.example.com", " x@y.org "])
def test_valid_emails(value):
    assert validate_field("email", value).valid


@pytest.mark.parametrize("value", [
    "plain", "a@b", "a@@b.com", "a b@c.com", "@b.com", "a@.com", "a..b@c.com", ".a@b.com",
    "a@b.c", "a@-b.com", 5,
])
def test_invalid_emails(value):
    bad("email", value)


def test_email_length_limit():
    assert validate_field("email", "a" * 60 + "@" + "d" * 60 + ".com").valid
    bad("email", "a" * 60 + "@" + ("d" * 60 + ".") * 4 + "com")  # 308 chars > 254


# --- degree ---
@pytest.mark.parametrize("raw,expected", [
    ("b tech", "B.Tech"), ("B.Tech", "B.Tech"), ("BTECH", "B.Tech"), ("b.e", "B.E."), ("BE", "B.E."),
    ("b.sc", "B.Sc"), ("bca", "BCA"), ("MBA Tech", "MBA Tech"), ("m tech", "M.Tech"),
    ("MCA", "MCA"), ("other", "Other"), ("Bachelor of Technology", "B.Tech"),
])
def test_degree_normalisation(raw, expected):
    ok("degree", raw, expected)


@pytest.mark.parametrize("raw", ["BBA", "engineering", "", "PhD", 3])
def test_degree_invalid(raw):
    bad("degree", raw)


# --- graduation year ---
def test_graduation_year():
    ok("graduation_year", 2026, 2026)
    ok("graduation_year", 2028, 2028)
    ok("graduation_year", "2027", 2027)
    ok("graduation_year", 2027.0, 2027)
    for v in (2025, 2029, "27", "'27", "twenty twenty seven", 2027.5, True, "next year", "2k27"):
        bad("graduation_year", v)


# --- cgpa ---
@pytest.mark.parametrize("raw,expected", [
    (8.4, 8.4), ("8.4", 8.4), (8, 8.0), ("8", 8.0), (8.456, 8.46), (8.455, 8.46), ("9.1/10", 9.1),
    ("7.5 out of 10", 7.5), (0, 0.0), (10, 10.0), ("10.0", 10.0), (".5", 0.5), (9.999, 10.0),
])
def test_cgpa_normalisation(raw, expected):
    ok("cgpa", raw, expected)


@pytest.mark.parametrize("raw", [
    10.01, -0.1, 85, "85%", "3.5/4", "abc", "8,4", True, None, [8],
    float("nan"), float("inf"), "10.005",
])
def test_cgpa_invalid(raw):
    bad("cgpa", raw)


# --- location ---
@pytest.mark.parametrize("raw,expected", [
    ("Bangalore", "Bengaluru"), ("bengaluru", "Bengaluru"), ("MUMBAI", "Mumbai"), ("Bombay", "Mumbai"),
    ("remote", "Remote"), ("WFH", "Remote"), ("no preference", "No preference"),
    ("Anywhere", "No preference"), ("don't matter", "No preference"),
])
def test_location_normalisation(raw, expected):
    ok("preferred_location", raw, expected)


@pytest.mark.parametrize("raw", ["Delhi", "Pune", "", 4])
def test_location_invalid(raw):
    bad("preferred_location", raw)


# --- available_from ---
@pytest.mark.parametrize("raw,expected", [
    ("2026-06-01", "2026-06-01"), ("2026-01-01", "2026-01-01"), ("2027-12-31", "2027-12-31"),
    ("1 March 2027", "2027-03-01"), ("1st March, 2027", "2027-03-01"),
    ("March 1, 2027", "2027-03-01"), ("1-Mar-2027", "2027-03-01"), ("2027/3/1", "2027-03-01"),
    ("15/06/2026", "2026-06-15"), ("06/15/2026", "2026-06-15"), ("05/05/2026", "2026-05-05"),
    ("Sept 3 2026", "2026-09-03"),
])
def test_available_from_valid(raw, expected):
    ok("available_from", raw, expected)


@pytest.mark.parametrize("raw", [
    "01/02/2027",                     # day/month order ambiguous
    "March 2027",                     # no day
    "1 March",                        # no year
    "next month", "soon", "immediately", "1 Mar 27",
    "2025-12-31", "2028-01-01",       # out of range
    "2026-02-30", "2026-13-01",       # not real dates
    "", 20260601,
])
def test_available_from_invalid(raw):
    bad("available_from", raw)


def test_available_from_reasons_explain_gap():
    assert "ambiguous" in validate_field("available_from", "01/02/2027").reason
    assert "year is missing" in validate_field("available_from", "1 March").reason
    assert "day is missing" in validate_field("available_from", "March 2027").reason


# --- duration ---
@pytest.mark.parametrize("raw,expected", [
    (2, 2), (3, 3), (6, 6), ("3", 3), ("6 months", 6), ("two months", 2), ("three", 3), (3.0, 3),
])
def test_duration_valid(raw, expected):
    ok("internship_duration_months", raw, expected)


@pytest.mark.parametrize("raw", [0, 1, 4, 5, 12, "1 year", "four", "3.5", True, None])
def test_duration_invalid(raw):
    bad("internship_duration_months", raw)


# --- urls ---
def test_linkedin():
    ok("linkedin_url", "https://www.linkedin.com/in/aarav-shah", "https://www.linkedin.com/in/aarav-shah")
    ok("linkedin_url", "linkedin.com/in/aarav", "https://linkedin.com/in/aarav")
    ok("linkedin_url", "https://in.linkedin.com/in/aarav", "https://in.linkedin.com/in/aarav")
    for v in ("https://notlinkedin.com/in/a", "https://linkedin.com/", "https://evil.com/linkedin.com/in/a",
              "https://linkedin.com@evil.com/in/a", "ftp://linkedin.com/in/a", "not a url", 5):
        bad("linkedin_url", v)


def test_github():
    ok("github_url", "https://github.com/aarav-shah", "https://github.com/aarav-shah")
    ok("github_url", "github.com/aarav/", "https://github.com/aarav/")
    for v in ("https://github.com/", "https://github.com/user/repo", "https://gist.github.com/u",
              "https://notgithub.com/user", "https://github.com.evil.com/user",
              "https://github.com/-bad_name", 7):
        bad("github_url", v)


# --- skills ---
def test_skills():
    ok("skills", ["Python", " SQL "], ["Python", "SQL"])
    ok("skills", "Python, SQL,, Go", ["Python", "SQL", "Go"])
    ok("skills", ["s"] * 30, ["s"] * 30)
    ok("skills", ["x" * 50], ["x" * 50])
    for v in (["s"] * 31, ["x" * 51], [""], [1], [], "", {"a": 1}):
        bad("skills", v)


# --- tracking fields ---
def test_tracking_fields():
    ok("collection_status", "In Progress", "in_progress")
    bad("collection_status", "done")
    ok("attempt_count", 0, 0)
    ok("attempt_count", 50, 50)
    bad("attempt_count", 51)
    bad("attempt_count", -1)
    ok("last_contacted_at", "2026-11-20T15:30:00+05:30", "2026-11-20T15:30:00+05:30")
    bad("last_contacted_at", "2026-11-20T15:30:00")  # no timezone
    ok("notes", "x" * 4000, "x" * 4000)
    bad("notes", "x" * 4001)


# --- stored-value (strict) checks and record analysis ---
def test_check_stored_is_strict_about_form():
    assert check_stored("graduation_year", 2027).valid
    assert not check_stored("graduation_year", "2027").valid  # right value, wrong type
    assert not check_stored("degree", "btech").valid          # not canonical spelling
    assert check_stored("degree", "B.Tech").valid
    assert not check_stored("cgpa", 85).valid
    assert not check_stored("cgpa", "8.4").valid
    assert check_stored("cgpa", 8).valid
    assert not check_stored("cgpa", 8.456).valid
    assert check_stored("phone", "+919876543210").valid
    assert not check_stored("phone", "9876543210").valid


def good_record(**over):
    rec = {
        "id": "c900", "full_name": "Test Person", "phone": "+919876543210", "email": "t@example.com",
        "college": "Test College", "degree": "B.Tech", "branch": "Computer Engineering",
        "graduation_year": 2027, "cgpa": 8.4, "preferred_location": "Remote",
        "available_from": "2026-06-01", "internship_duration_months": 3,
        "collection_status": "pending", "attempt_count": 0, "version": 1,
    }
    rec.update(over)
    return rec


def test_analyse_complete_record():
    a = analyse_record(good_record())
    assert not a.missing and not a.invalid and not a.needs_collection
    assert a.collection_status == "pending" and a.attempt_count == 0


def test_analyse_missing_and_invalid():
    a = analyse_record(good_record(email=None, cgpa="", graduation_year=2035, degree="BBA"))
    assert a.missing == ["email", "cgpa"]
    assert set(a.invalid) == {"graduation_year", "degree"}
    assert a.fields_to_collect == ["email", "degree", "graduation_year", "cgpa"]


def test_invalid_optional_does_not_block_completion():
    a = analyse_record(good_record(linkedin_url="https://evil.com/x", skills=["Python"]))
    assert a.optional_populated == ["linkedin_url", "skills"]
    assert set(a.optional_invalid) == {"linkedin_url"}
    assert not a.needs_collection


def test_missing_phone_reported_but_not_collectable():
    a = analyse_record(good_record(phone=None))
    assert a.missing == ["phone"] and a.blocked_read_only == ["phone"]
    assert a.fields_to_collect == [] and not a.needs_collection
