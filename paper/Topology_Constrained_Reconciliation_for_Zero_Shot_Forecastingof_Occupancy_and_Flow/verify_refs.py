"""Check every bibliography entry against Crossref, arXiv or DataCite.

The ground rule for this project is that nothing goes into the paper on the strength of
a recollection, and a citation is no exception: a plausible-looking volume and page range
is exactly the kind of thing that is wrong half the time and that a reader cannot check
without the lookup we should have done. Twenty-three entries carried a ``% VERIFY``
marker, meaning they were written from memory and never checked. This script checks them.

It prints, for each entry, the recorded metadata beside whatever the registry returns, and
flags the fields that disagree. It does not edit the file: the corrections are made by
hand so that each one is a deliberate decision, and a title that cannot be found at all is
a different problem from a title whose volume is off by one.

Usage::

    python verify_refs.py            # every entry
    python verify_refs.py --todo     # only the entries still marked % VERIFY
"""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

BIB = Path(__file__).resolve().parent / "ref.bib"
CROSSREF = "https://api.crossref.org/works"
ARXIV = "http://export.arxiv.org/api/query"
USER_AGENT = "mflow-ref-check/1.0 (mailto:shb@3sholding.com)"

#: Fields worth comparing. Author is compared on surnames only, because registries
#: disagree harmlessly about initials and accents.
COMPARED = ("title", "journal", "year", "volume", "number", "pages", "author")


class VerificationError(RuntimeError):
    """Raised when the bibliography cannot be parsed, rather than silently skipping."""


def fetch(url: str, params: dict[str, str]) -> str:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{url}?{query}", headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


# --------------------------------------------------------------------------- #
# Parsing the bib file
# --------------------------------------------------------------------------- #


def parse_bib(text: str) -> list[dict[str, Any]]:
    """Split the file into entries. Deliberately simple: this file is hand-written."""
    entries: list[dict[str, Any]] = []
    for match in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", text, flags=re.S):
        kind, key, body = match.group(1), match.group(2).strip(), match.group(3)
        fields: dict[str, str] = {}
        for field in re.finditer(r"(\w+)\s*=\s*\{(.*?)\}\s*(?:,|$)", body, flags=re.S):
            fields[field.group(1).lower()] = " ".join(field.group(2).split())
        entries.append(
            {
                "key": key,
                "kind": kind,
                "fields": fields,
                "unverified": "% VERIFY" in text[match.end() : match.end() + 40],
            }
        )
    if not entries:
        raise VerificationError(f"no entries parsed out of {BIB}")
    return entries


def fold(value: str) -> str:
    """Lowercase and strip accents, so that Koster and K\u00f6ster compare equal.

    Both sides need this. Registries return composed Unicode and the bib file carries
    LaTeX escapes, and without folding every accented surname in the file reports as a
    disagreement, which buries the handful that are real.
    """
    value = re.sub(r"\\[`'^\"~=.]", "", value)
    value = unicodedata.normalize("NFKD", value)
    value = "".join(c for c in value if not unicodedata.combining(c))
    return re.sub(r"[\\'\"`^~{}]", "", value).lower()


def surnames(author: str) -> list[str]:
    """Surnames from an author field, folded, with name particles dropped.

    Crossref splits "Di Fonzo" into a given name "Tommaso Di" and a family name "Fonzo",
    so comparing the last token on both sides is the only stable choice.
    """
    out: list[str] = []
    for person in author.split(" and "):
        person = person.strip()
        if not person:
            continue
        name = person.split(",")[0] if "," in person else person
        tokens = fold(name).split()
        if tokens:
            out.append(tokens[-1])
    return out


def normalise(value: str) -> str:
    value = fold(value)
    value = value.replace("&amp;", "and").replace("&", "and")
    value = value.replace("\u2013", "-").replace("--", "-")
    return " ".join(re.sub(r"[^a-z0-9\- ]", " ", value).split())


def same_title(ours: str, theirs: str) -> bool:
    """Whether two titles are the same work.

    Containment alone is too weak and produced wrong matches: the title "Convex
    Optimization" is contained in "CVXPY: a Python-embedded modeling language for convex
    optimization", so a book chapter was accepted as a match for the CVXPY paper. Requiring
    most of the words of the *longer* title to appear in the shorter one rules that out
    while still tolerating a dropped subtitle.
    """
    left, right = set(normalise(ours).split()), set(normalise(theirs).split())
    if not left or not right:
        return False
    return len(left & right) >= 0.75 * max(len(left), len(right))


# --------------------------------------------------------------------------- #
# Registry lookups
# --------------------------------------------------------------------------- #


def crossref_by_doi(doi: str) -> dict[str, Any] | None:
    try:
        raw = fetch(f"{CROSSREF}/{urllib.parse.quote(doi)}", {})
    except Exception:  # noqa: BLE001
        return None
    return json.loads(raw).get("message")


def crossref_by_title(title: str) -> dict[str, Any] | None:
    try:
        raw = fetch(CROSSREF, {"query.bibliographic": title, "rows": "5"})
    except Exception:  # noqa: BLE001
        return None
    items = json.loads(raw).get("message", {}).get("items", [])
    for item in items:
        found = " ".join(item.get("title", []) or [])
        if found and same_title(title, found):
            return item
    return None


def arxiv_by_title(title: str) -> dict[str, Any] | None:
    try:
        raw = fetch(ARXIV, {"search_query": f'ti:"{title}"', "max_results": "3"})
    except Exception:  # noqa: BLE001
        return None
    for block in re.findall(r"<entry>(.*?)</entry>", raw, flags=re.S):
        found_title = re.search(r"<title>(.*?)</title>", block, flags=re.S)
        if not found_title or not same_title(title, found_title.group(1)):
            continue
        identifier = re.search(r"<id>http://arxiv.org/abs/([^<]+)</id>", block)
        authors = re.findall(r"<name>([^<]+)</name>", block)
        published = re.search(r"<published>(\d{4})", block)
        return {
            "source": "arxiv",
            "arxiv": identifier.group(1) if identifier else "",
            "title": [found_title.group(1).strip()],
            "author": [{"family": a.split()[-1]} for a in authors],
            "year": published.group(1) if published else "",
        }
    return None


def registry_record(fields: dict[str, str]) -> tuple[dict[str, Any] | None, str]:
    if fields.get("doi"):
        record = crossref_by_doi(fields["doi"])
        if record is not None:
            return record, "crossref/doi"
    title = fields.get("title", "")
    if not title:
        return None, "no title"
    record = crossref_by_title(title)
    if record is not None:
        return record, "crossref/title"
    record = arxiv_by_title(title)
    if record is not None:
        return record, "arxiv/title"
    return None, "not found"


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #


def record_fields(record: dict[str, Any]) -> dict[str, str]:
    if record.get("source") == "arxiv":
        return {
            "title": " ".join(record["title"]),
            "year": str(record.get("year", "")),
            "author": " and ".join(a["family"] for a in record.get("author", [])),
            "arxiv": record.get("arxiv", ""),
        }
    container = record.get("container-title") or []
    # Crossref's "issued" is the earliest registered date, which for a journal that
    # posts online first is a year before the issue the article is cited by. Prefer the
    # print date when there is one, since that is what a reference list carries.
    years: list[str] = []
    for field in ("published-print", "issued", "published"):
        parts = record.get(field, {}).get("date-parts") or [[None]]
        if parts and parts[0] and parts[0][0]:
            years.append(str(parts[0][0]))
    return {
        "title": " ".join(record.get("title") or []),
        "journal": container[0] if container else "",
        "year": years[0] if years else "",
        "year_alternatives": ",".join(dict.fromkeys(years)),
        "volume": str(record.get("volume", "")),
        "number": str(record.get("issue", "")),
        "pages": str(record.get("page", "")),
        "author": " and ".join(
            a.get("family", "") for a in record.get("author", []) if a.get("family")
        ),
        "doi": record.get("DOI", ""),
    }


def disagreements(mine: dict[str, str], theirs: dict[str, str]) -> list[str]:
    out: list[str] = []
    for field in COMPARED:
        ours = mine.get(field, "")
        yours = theirs.get(field, "")
        if not ours or not yours:
            continue
        if field == "author":
            if set(surnames(ours)) != set(surnames(yours)):
                out.append(f"{field}: ours {surnames(ours)} vs theirs {surnames(yours)}")
            continue
        if field == "year":
            if ours not in (theirs.get("year_alternatives") or yours).split(","):
                out.append(f"{field}: ours {ours!r} vs theirs {yours!r}")
            continue
        if field == "pages":
            # Crossref often records only the first page of an article.
            if normalise(ours).split("-")[0] != normalise(yours).split("-")[0]:
                out.append(f"{field}: ours {ours!r} vs theirs {yours!r}")
            continue
        if normalise(ours) != normalise(yours):
            out.append(f"{field}: ours {ours!r} vs theirs {yours!r}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--todo", action="store_true", help="only entries still marked %% VERIFY"
    )
    args = parser.parse_args()

    entries = parse_bib(BIB.read_text(encoding="utf-8"))
    if args.todo:
        entries = [e for e in entries if e["unverified"]]
    print(f"checking {len(entries)} entries against Crossref and arXiv\n")

    clean, mismatched, missing = [], [], []
    for entry in entries:
        key, fields = entry["key"], entry["fields"]
        record, how = registry_record(fields)
        time.sleep(0.4)
        if record is None:
            missing.append(key)
            print(f"NOT FOUND  {key:28s} {fields.get('title', '')[:70]!r}")
            continue
        problems = disagreements(fields, record_fields(record))
        if problems:
            mismatched.append(key)
            print(f"MISMATCH   {key:28s} via {how}")
            for problem in problems:
                print(f"           {problem}")
            theirs = record_fields(record)
            if theirs.get("doi"):
                print(f"           doi: {theirs['doi']}")
            if theirs.get("arxiv"):
                print(f"           arXiv: {theirs['arxiv']}")
        else:
            clean.append(key)
            print(f"ok         {key:28s} via {how}")

    print(
        f"\n{len(clean)} agree, {len(mismatched)} disagree, {len(missing)} not found"
    )
    if mismatched:
        print("disagree: " + ", ".join(mismatched))
    if missing:
        print("not found: " + ", ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
