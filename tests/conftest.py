"""Shared pytest collection policy.

Distributed integration and validation modules coordinate their own ranks from
``__main__`` and must not be collected as independent pytest functions.
"""

collect_ignore_glob = [
    "integration/**/test_*.py",
    "validation/**/test_*.py",
]
