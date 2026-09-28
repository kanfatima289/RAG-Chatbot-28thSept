"""Phase 1 gate, item C9 - the Groq key lives in .env and never in the repo.

These are cheap tests and they protect the one secret in the project, so they
assert on the *files* as well as on the loaded values. A test that only reads
`config.GROQ_API_KEY` would still pass if the key were committed in a
different file, which is the failure that actually happens.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

import config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"
ENV = PROJECT_ROOT / ".env"
GITIGNORE = PROJECT_ROOT / ".gitignore"

#: A real key starts with gsk_. Cheap, and it is the thing worth catching: a
#: committed key usually looks exactly like this.
KEY_SHAPED = re.compile(r"\bgsk_[A-Za-z0-9]{20,}")

#: One entry per line, `#` comments and blanks ignored. `.env.example` is full
#: of comments, so a regex has to be anchored to the start of a line and skip
#: them - otherwise "GROQ_API_KEY= " matches the `#` of the next line.
_ENV_LINE = re.compile(r"(?m)^([A-Z][A-Z0-9_]*)[ \t]*=[ \t]*(.*?)[ \t]*$")


def _read_env(path: Path) -> dict[str, str]:
    """Parse a dotenv file into a dict, ignoring comments and blanks."""
    values: dict[str, str] = {}
    for name, raw in _ENV_LINE.findall(path.read_text("utf-8")):
        value = raw.strip()
        if value[:1] in {"'", '"'}:  # strip a quoted value
            value = value[1:-1] if value[-1:] == value[:1] and len(value) > 1 else value[1:]
        values[name] = value
    return values

#: Directories that must never be committed regardless of .env handling.
FORBIDDEN_TRACKED = (".env", "venv/", ".venv/", "__pycache__/", "data/chroma/")


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result.stdout


# ---------------------------------------------------------------------------
# .env.example exists and documents both variables
# ---------------------------------------------------------------------------
def test_env_example_exists() -> None:
    assert ENV_EXAMPLE.is_file(), ".env.example is missing"


def test_env_example_is_committed() -> None:
    """The template has to be tracked, or a fresh clone cannot be configured."""
    assert ".env.example" in _git("ls-files").splitlines()


@pytest.mark.parametrize("name", ["GROQ_API_KEY", "GROQ_MODEL"])
def test_env_example_defines_the_variable(name: str) -> None:
    values = _read_env(ENV_EXAMPLE)
    assert name in values, f"{name} is not defined in .env.example"
    if name == "GROQ_API_KEY":
        return  # deliberately empty; asserted separately
    assert values[name], f"{name} is present but has no value"


def test_env_example_carries_no_key() -> None:
    """The template must ship an empty key, or it ships someone's key."""
    text = ENV_EXAMPLE.read_text("utf-8")
    assert not KEY_SHAPED.search(text), ".env.example contains something key-shaped"
    assert _read_env(ENV_EXAMPLE)["GROQ_API_KEY"] == "", (
        "GROQ_API_KEY in .env.example must be empty"
    )


def test_env_example_groq_model_is_a_real_model_id() -> None:
    model = _read_env(ENV_EXAMPLE)["GROQ_MODEL"]
    # Verified to answer this corpus on the current key (phase 5 live run):
    # an OpenAI open-model or a Qwen family id. Anything else the key later
    # permits can be added when it has been measured here, not before.
    assert model.startswith(("llama-", "openai/gpt-oss", "qwen/")), (
        f"unexpected GROQ_MODEL {model!r}"
    )


def test_env_example_keeps_every_tunable_in_step_with_config() -> None:
    """`.env` and `config.py` disagreeing silently is a real past bug.

    `load_dotenv()` runs before the defaults are read, so a stale `.env`
    overrides an edit to `config.py` without any warning - that is how
    `CHUNK_SIZE=400` survived a config change to 250 during phase 3.
    """
    example = _read_env(ENV_EXAMPLE)
    for name in ("EMBED_MODEL", "EMBED_MAX_TOKENS", "EMBED_BACKEND",
                 "EMBED_BATCH_SIZE", "TOP_K", "MIN_SCORE",
                 "CHUNK_SIZE", "CHUNK_OVERLAP", "MIN_SIZE"):
        assert name in example, f"{name} missing from .env.example"
        assert str(getattr(config, name)) == example[name], (
            f"{name} is {example[name]!r} in .env.example but "
            f"{getattr(config, name)!r} in config.py - one is stale"
        )


# ---------------------------------------------------------------------------
# .env is ignored, and nothing secret is tracked
# ---------------------------------------------------------------------------
def test_gitignore_mentions_env() -> None:
    text = GITIGNORE.read_text("utf-8")
    assert re.search(r"(?m)^\.env\s*$", text), ".env is not on its own gitignore line"


def test_env_is_ignored_by_git() -> None:
    """`git check-ignore` is the authoritative answer, not the file contents."""
    result = subprocess.run(
        ["git", "check-ignore", "-v", ".env"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, ".env is not ignored by git"


def test_env_is_not_tracked() -> None:
    assert ".env" not in _git("ls-files").splitlines(), (
        ".env is tracked by git - the key is committed"
    )


@pytest.mark.parametrize("path", FORBIDDEN_TRACKED)
def test_generated_and_secret_paths_are_untracked(path: str) -> None:
    """`path` is matched exactly, not by prefix.

    A prefix match says ".env.example" is a violation of ".env", which is
    backwards: the template is the one file here that must be committed.
    """
    tracked = _git("ls-files").splitlines()
    offenders = [t for t in tracked if t == path or t.startswith(path.rstrip("/") + "/")]
    assert not offenders, f"tracked but must not be: {offenders}"


def test_no_tracked_file_contains_a_key() -> None:
    """The check that would have caught a leaked key after the fact."""
    offenders = []
    for rel in _git("ls-files").splitlines():
        path = PROJECT_ROOT / rel
        if not path.is_file():
            continue
        try:
            text = path.read_text("utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            # Skip the regexes and tests that look for keys on purpose.
            if "gsk_" not in line and "KEY_SHAPED" not in line:
                continue
            if line.lstrip().startswith("#"):
                continue
            stripped = line.strip()
            if stripped.startswith(('r"', "r'", '"gsk_', "'gsk_")):
                continue
            if re.match(r"^(assert|assert not|.*KEY_SHAPED\s*=|.*re\.compile)", stripped):
                continue
            if "gsk_" in stripped and re.search(r"gsk_[A-Za-z0-9]{20,}", stripped):
                offenders.append(f"{rel}:{lineno}")
    assert not offenders, f"possible committed key at: {offenders}"


# ---------------------------------------------------------------------------
# python-dotenv actually loads
# ---------------------------------------------------------------------------
def test_python_dotenv_is_a_declared_dependency() -> None:
    reqs = (PROJECT_ROOT / "requirements.txt").read_text("utf-8")
    assert "python-dotenv" in reqs, "python-dotenv is not in requirements.txt"


def test_python_dotenv_is_installed() -> None:
    import dotenv

    assert callable(dotenv.load_dotenv)


def test_config_calls_load_dotenv_before_reading_defaults() -> None:
    """Order matters and is not obvious from reading the constants.

    `load_dotenv()` must be called before the `os.getenv` defaults, or `.env`
    has no effect at all.
    """
    text = (PROJECT_ROOT / "config.py").read_text("utf-8")
    assert "load_dotenv()" in text, "config.py never calls load_dotenv()"
    assert text.index("load_dotenv()") < text.index("os.getenv"), (
        "load_dotenv() runs after the first os.getenv, so .env cannot override it"
    )


def test_groq_model_is_loaded() -> None:
    assert config.GROQ_MODEL
    assert os_environ_matches_file("GROQ_MODEL", config.GROQ_MODEL)


def test_missing_key_is_a_banner_not_a_crash() -> None:
    """C13. The app must start and explain, not raise on import."""
    assert isinstance(config.has_groq_key(), bool)
    assert config.has_groq_key() is bool(config.GROQ_API_KEY.strip())


def test_local_env_key_is_never_tracked() -> None:
    """A key set in the local `.env` is the deliberate live state (phase 5+).

    The invariant is not "no key exists" - the demo needs one - it is that a
    key can never be committed. `test_env_is_ignored_by_git` proves git
    ignores `.env` outright; this closes the one remaining route, a forced
    `git add`, by checking a key-shaped value never lands in tracked content.
    """
    if not ENV.is_file():
        pytest.skip("no local .env")
    value = re.search(r"(?m)^GROQ_API_KEY\s*=\s*(\S*)", ENV.read_text("utf-8"))
    if value is None or not KEY_SHAPED.search(value.group(1)):
        pytest.skip("no key set in local .env - nothing to guard")
    # The key exists locally: prove git cannot take it, both ways.
    result = subprocess.run(
        ["git", "check-ignore", "-v", ".env"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, ".env is not ignored by git"
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", ".env"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert status.stdout.strip() == "", (
        f"tracked .env changes: {status.stdout.strip()!r} - a key is being staged"
    )


def os_environ_matches_file(name: str, value: str) -> bool:
    import os

    return os.environ.get(name) == value
