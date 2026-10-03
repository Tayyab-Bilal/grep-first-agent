"""Every ```python block in README.md and docs/*.md must run. Docs that rot fail the build."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FILES = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)
CASES = [(f, i, code) for f in FILES for i, code in enumerate(BLOCK.findall(f.read_text()))]


def test_docs_have_runnable_examples():
    assert len(CASES) >= 8


@pytest.mark.parametrize("path,index,code", CASES, ids=[f"{f.name}[{i}]" for f, i, _ in CASES])
def test_doc_snippet_runs(path, index, code):
    exec(compile(code, f"{path.name}[{index}]", "exec"), {"__name__": "__doc_snippet__"})  # noqa: S102 - running our own docs
