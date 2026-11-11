"""The README's numbers are derived from the code, and the checker has teeth.

``scripts/check_readme_claims.py`` runs in CI's lint job. These tests make
sure it would actually fail on a wrong number, a reworded claim, or a mutation
table that has drifted from the mutation list, because a checker that cannot
fail is a checker that only adds confidence it has not earned.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_checker():
    path = ROOT / "scripts" / "check_readme_claims.py"
    spec = importlib.util.spec_from_file_location("_check_readme_claims", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = _load_checker()
README = (ROOT / "README.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def truths() -> dict[str, str]:
    return checker.gather_truths()


def test_readme_numbers_match_the_code(truths) -> None:
    mismatches, missing = checker.check(README, truths)
    assert not mismatches, "\n".join(str(m) for m in mismatches)
    assert not missing, [c.pattern for c in missing]
    names = [m.name for m in checker.load_mutations()]
    assert not checker.check_mutation_table(README, names)
    assert not checker.check_versions(truths["version"])


def test_checker_notices_a_wrong_number(truths) -> None:
    """Bump the test count in one place: the checker must name that line."""
    match = re.search(r"(\d+) tests across world sizes", README)
    assert match is not None
    wrong = str(int(match.group(1)) + 1)
    text = README[: match.start(1)] + wrong + README[match.end(1) :]
    line = README.count("\n", 0, match.start()) + 1

    mismatches, _ = checker.check(text, truths)

    assert len(mismatches) == 1, [str(m) for m in mismatches]
    report = str(mismatches[0])
    assert report.startswith(f"README.md:{line}:"), report
    assert repr(wrong) in report and repr(truths["tests"]) in report, report


def test_checker_reads_number_words(truths) -> None:
    """'nine-member' is a claim too, written as a word."""
    text = README.replace("nine-member", "eight-member", 1)
    mismatches, _ = checker.check(text, truths)
    assert [m.claimed for m in mismatches] == ["eight"], [str(m) for m in mismatches]


def test_checker_reports_a_claim_that_vanished(truths) -> None:
    """Rewording a sentence must not quietly stop it being checked."""
    text = re.sub(r"Current\s+release:", "Latest:", README)
    _, missing = checker.check(text, truths)
    assert [c.truth for c in missing] == ["version"]


def test_mutation_table_must_list_every_mutation() -> None:
    names = [m.name for m in checker.load_mutations()]
    problems = checker.check_mutation_table(README, [*names, "imaginary-mutation"])
    assert problems == ["README mutation table is missing `imaginary-mutation`"]
    problems = checker.check_mutation_table(README, names[1:])
    assert problems == [
        f"README mutation table lists `{names[0]}`, which is not a real mutation"
    ]
