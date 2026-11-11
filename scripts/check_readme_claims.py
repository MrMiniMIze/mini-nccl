"""Check that the numbers the README states still match the code.

    python scripts/check_readme_claims.py

The README makes countable claims: how many tests there are, how many modules
mypy checks, how many injected defects the mutation checker catches, how many
members the ``Communicator`` protocol has, and which version is current. Those
numbers were hand-edited, and they drifted (two commits in this history exist
only to correct them). For a project whose pitch is that every claim is
checkable, that is the wrong way round, so this derives each number from its
source of truth and compares.

It never rewrites the README. A mismatch is reported with the line it is on and
both values, and a human fixes the prose, because the sentence around a number
usually needs to change with it.

Every claim must also be *found*. If the README is reworded so a pattern no
longer matches, that is reported too: otherwise rewording a sentence would
quietly stop it being checked, which is the same rot ``mutation_check.py
--strict`` guards against for its anchors.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}


@dataclass(frozen=True)
class Claim:
    """A pattern in the README whose ``n`` group must equal a derived fact."""

    pattern: str
    truth: str


@dataclass(frozen=True)
class Mismatch:
    line: int
    excerpt: str
    claimed: str
    actual: str
    source: str

    def __str__(self) -> str:
        return (
            f"README.md:{self.line}: says {self.claimed!r} in {self.excerpt!r}, "
            f"but {self.source} gives {self.actual!r}"
        )


# Whitespace in the patterns is \s+ so a claim still matches when the prose
# wraps it across lines.
CLAIMS: list[Claim] = [
    Claim(r"Current\s+release:\s+\*\*v(?P<n>\d+\.\d+\.\d+)\*\*", "version"),
    Claim(r"(?P<n>\d+)\s+tests\s+across\s+world\s+sizes", "tests"),
    Claim(r"#\s+(?P<n>\d+)\s+tests,", "tests"),
    Claim(r"clean\s+across\s+(?P<n>\d+)\s+modules", "modules"),
    Claim(r"across\s+all\s+(?P<n>\d+)\s+modules", "modules"),
    Claim(r"(?P<n>\d+)\s+of\s+\d+\s+injected\s+defects", "mutations"),
    Claim(r"\d+\s+of\s+(?P<n>\d+)\s+injected\s+defects", "mutations"),
    Claim(r"all\s+(?P<n>\d+)\s+real\s+mutations\s+are\s+caught", "mutations"),
    Claim(r"\((?P<n>\d+)\s+on\s+a\s+GPU-less\s+runner", "mutations_without_gpu"),
    Claim(r"(?P<n>\w+)-member\s+`Protocol`", "protocol_members"),
    Claim(r"written\s+against\s+(?P<n>\w+)\s+members", "protocol_members"),
    Claim(r"naming\s+the\s+(?P<n>\w+)\s+members", "protocol_members"),
]

SOURCES = {
    "version": "mini_nccl/__init__.py",
    "tests": "pytest --collect-only",
    "modules": "mini_nccl/*.py",
    "mutations": "scripts/mutation_check.py (real mutations)",
    "mutations_without_gpu": "scripts/mutation_check.py (real, not needing CUDA)",
    "protocol_members": "Communicator in mini_nccl/communicator.py",
}


def _normalize(value: str) -> str:
    return str(_WORDS.get(value.lower(), value))


def check(text: str, truths: dict[str, str]) -> tuple[list[Mismatch], list[Claim]]:
    """Compare every claim in ``text`` with ``truths``.

    Returns the mismatches, and the claims whose pattern matched nowhere.
    """
    mismatches: list[Mismatch] = []
    missing: list[Claim] = []
    for claim in CLAIMS:
        matches = list(re.finditer(claim.pattern, text))
        if not matches:
            missing.append(claim)
            continue
        for match in matches:
            claimed = _normalize(match.group("n"))
            actual = str(truths[claim.truth])
            if claimed != actual:
                mismatches.append(
                    Mismatch(
                        line=text.count("\n", 0, match.start()) + 1,
                        excerpt=" ".join(match.group(0).split()),
                        claimed=match.group("n"),
                        actual=actual,
                        source=SOURCES[claim.truth],
                    )
                )
    return mismatches, missing


def check_mutation_table(text: str, names: list[str]) -> list[str]:
    """The README's mutation table must list exactly the real mutations."""
    section = text.split("### Do the tests have teeth?", 1)[-1]
    listed = re.findall(r"^\| `([a-z0-9-]+)` \|", section, flags=re.MULTILINE)
    problems = []
    for name in sorted(set(names) - set(listed)):
        problems.append(f"README mutation table is missing `{name}`")
    for name in sorted(set(listed) - set(names)):
        problems.append(f"README mutation table lists `{name}`, which is not a real mutation")
    return problems


def check_versions(version: str) -> list[str]:
    """pyproject must take its version from the package, and the changelog agree."""
    problems = []
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    single_sourced = re.search(
        r'version\s*=\s*\{\s*attr\s*=\s*"mini_nccl\.__version__"\s*\}', pyproject
    )
    if single_sourced is None or re.search(r'^version\s*=\s*"', pyproject, re.M):
        problems.append(
            "pyproject.toml must read its version from mini_nccl.__version__ "
            "(single source of truth)"
        )
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    released = re.search(r"^## (\d+\.\d+\.\d+)", changelog, flags=re.MULTILINE)
    if released is None or released.group(1) != version:
        found = released.group(1) if released else "none"
        problems.append(
            f"CHANGELOG.md's newest release is {found}, but the package is {version}"
        )
    return problems


# ---- sources of truth ---------------------------------------------------------


def read_version() -> str:
    tree = ast.parse((ROOT / "mini_nccl" / "__init__.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets)
            and isinstance(node.value, ast.Constant)
        ):
            return str(node.value.value)
    raise RuntimeError("no __version__ literal in mini_nccl/__init__.py")


def count_tests() -> int:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    found = re.search(r"(\d+) tests? collected", proc.stdout)
    if proc.returncode != 0 or found is None:
        raise RuntimeError(f"test collection failed:\n{proc.stdout}\n{proc.stderr}")
    return int(found.group(1))


def count_modules() -> int:
    return len(list((ROOT / "mini_nccl").glob("*.py")))


def load_mutations() -> list:
    path = ROOT / "scripts" / "mutation_check.py"
    spec = importlib.util.spec_from_file_location("_mutation_check", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before execution so its dataclasses can resolve their module.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return [m for m in module.MUTATIONS if not m.expect_survival]


def count_protocol_members() -> int:
    tree = ast.parse((ROOT / "mini_nccl" / "communicator.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Communicator":
            return sum(
                isinstance(item, (ast.AnnAssign, ast.FunctionDef)) for item in node.body
            )
    raise RuntimeError("no Communicator class in mini_nccl/communicator.py")


def gather_truths() -> dict[str, str]:
    mutations = load_mutations()
    return {
        "version": read_version(),
        "tests": str(count_tests()),
        "modules": str(count_modules()),
        "mutations": str(len(mutations)),
        "mutations_without_gpu": str(sum(m.requires is None for m in mutations)),
        "protocol_members": str(count_protocol_members()),
    }


def main() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    truths = gather_truths()
    mismatches, missing = check(text, truths)
    problems = [str(m) for m in mismatches]
    problems += [
        f"README.md: no longer matches the claim pattern {c.pattern!r} "
        f"(reworded? update CLAIMS in this script)"
        for c in missing
    ]
    problems += check_mutation_table(text, [m.name for m in load_mutations()])
    problems += check_versions(truths["version"])

    if problems:
        print("README claims that do not match the code:\n")
        for problem in problems:
            print(f"  {problem}")
        raise SystemExit(1)
    facts = ", ".join(f"{key}={value}" for key, value in truths.items())
    print(f"README claims match the code ({facts}).")


if __name__ == "__main__":
    main()
