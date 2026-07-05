"""Research-paper updater (Phase 2, P2.6).

Appends a completed study's rendered result section to the living research paper
(``docs/RESEARCH_FINDINGS.md``). The paper is a living deliverable — each study's
real, cost-inclusive, kill-gate numbers are written as part of the study, never
pre-filled with imagined results (Rules of Engagement).
"""

from __future__ import annotations

from pathlib import Path

from lab.research.reports.report import StudyReport, render_report

DEFAULT_FINDINGS_PATH = Path("docs") / "RESEARCH_FINDINGS.md"


def append_study_section(
    report: StudyReport, *, findings_path: Path = DEFAULT_FINDINGS_PATH
) -> None:
    """Append ``report``'s rendered section to the research paper.

    The file must already exist (the scaffold is created before Phase 3). The
    section is appended with a leading separator so studies accumulate in order.
    """
    if not findings_path.exists():
        raise FileNotFoundError(f"research paper not found: {findings_path}")
    section = render_report(report)
    with findings_path.open("a", encoding="utf-8") as handle:
        handle.write("\n---\n\n" + section + "\n")
