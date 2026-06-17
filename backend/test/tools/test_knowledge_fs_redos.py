"""Regression tests for ReDoS hardening of model-supplied knowledge-base grep patterns.

The knowledge-base filesystem tools (``grep_knowledge_files`` and ``kb_exec``'s
``grep``) build a matcher from a model-controlled ``pattern`` via
``knowledge_fs.build_matcher`` and run it line-by-line over knowledge-file
content. Without a guard, a pattern such as ``([a-z]+)+$`` against a modestly
long non-matching line triggers catastrophic backtracking and burns CPU with no
time bound.

These tests load ``knowledge_fs`` directly by path (so they do not pull in the
``open_webui`` package ``__init__``, which imports heavy CLI deps) and assert
that:

* known catastrophic patterns are rejected at build time, and
* even if a matcher is produced, a single scan stays well under a hard wall-clock
  bound — so the test itself can never hang.
"""

import importlib.util
import os
import time

import pytest

# Load backend/open_webui/tools/knowledge_fs.py in isolation.
_KFS_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'open_webui', 'tools', 'knowledge_fs.py')
)
_spec = importlib.util.spec_from_file_location('knowledge_fs_under_test', _KFS_PATH)
knowledge_fs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(knowledge_fs)


# Catastrophic-backtracking patterns that the regex auto-detection promotes to
# regex mode (each contains a bracket class / \d / \w / .* marker).
CATASTROPHIC_PATTERNS = [
    '([a-z]+)+$',
    '([a-z]+)*$',
    '([a-z]*)*$',
    '(\\d+)+$',
    '(\\w+)*!',
    '(.*a)*$',
    '(\\s+)+x',
]

# Catastrophic patterns that contain no regex marker, so they are only reachable
# in regex mode when the caller forces it (grep -E / use_regex=True).
CATASTROPHIC_PATTERNS_REGEX_FORCED = [
    '(a+)+$',
    '(ab+)+c',
]

# Legitimate regex patterns that must keep working unchanged.
LEGITIMATE_PATTERNS = [
    'error|warn',
    'version \\d+',
    '[a-z]+',
    '\\d{3}-\\d{4}',
    'foo.*bar',
    'user_[0-9]+',
    '(cat|dog)s?',
    '\\w+@\\w+',
    '(\\d{1,3}\\.){3}',
    '(\\w{2,5})+',
]


@pytest.mark.parametrize('pattern', CATASTROPHIC_PATTERNS)
def test_catastrophic_patterns_are_rejected(pattern):
    """Auto-promoted nested-quantifier patterns are refused with a clear error."""
    matcher, err = knowledge_fs.build_matcher(pattern)
    assert matcher is None
    assert err is not None
    assert 'backtracking' in err.lower()


@pytest.mark.parametrize('pattern', CATASTROPHIC_PATTERNS_REGEX_FORCED)
def test_catastrophic_patterns_rejected_when_regex_forced(pattern):
    """The grep -E / use_regex=True path also rejects nested-quantifier patterns."""
    matcher, err = knowledge_fs.build_matcher(pattern, use_regex=True)
    assert matcher is None
    assert err is not None
    assert 'backtracking' in err.lower()


@pytest.mark.parametrize('pattern', LEGITIMATE_PATTERNS)
def test_legitimate_patterns_still_compile(pattern):
    """Ordinary regex patterns continue to build a working matcher."""
    matcher, err = knowledge_fs.build_matcher(pattern)
    assert err is None
    assert matcher is not None
    # Smoke-check it is callable on representative input.
    assert matcher('error: version 12 user_7 cat dog a@b 10.0.0.1') in (True, False)


def test_grep_scan_is_time_bounded_for_evil_pattern():
    """A full line-by-line scan with a hostile pattern stays well under a hard bound.

    The matcher is obtained directly (mirroring how grep_knowledge_files /
    _kb_grep call build_matcher) and exercised over many catastrophic lines, the
    way a grep over knowledge-file content would. The whole loop must finish
    quickly; the assertion's generous ceiling means the test can never hang even
    if the guard regresses to the old unbounded behaviour (it would simply fail).
    """
    # If the pattern is rejected outright, the scan is trivially bounded.
    matcher, err = knowledge_fs.build_matcher('([a-z]+)+$')
    if matcher is None:
        assert err is not None
        return

    evil_lines = ['a' * 50 + '!'] * 200  # each line is individually catastrophic
    start = time.monotonic()
    for line in evil_lines:
        matcher(line)
    elapsed = time.monotonic() - start

    # The wall-clock backstop (REGEX_TIME_BUDGET_SECONDS) plus the per-line cap
    # must keep this far below the many-seconds-per-line blowup of the old code.
    assert elapsed < 5.0, f'scan took {elapsed:.2f}s — ReDoS guard not effective'


def test_per_line_length_cap_bounds_single_line():
    """A single very long hostile line is bounded by the per-line length cap."""
    matcher, err = knowledge_fs.build_matcher('foo.*bar')  # benign regex, exercises wrapper
    assert err is None
    long_line = 'x' * (knowledge_fs.MAX_REGEX_LINE_CHARS * 4)
    start = time.monotonic()
    matcher(long_line)
    assert time.monotonic() - start < 2.0
