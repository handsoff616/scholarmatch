"""Automated UI regression tests for ScholarMatch Streamlit dashboard using AppTest."""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def test_ui_initial_load():
    """Verify that the Streamlit app loads with no exceptions and initializes all tabs."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert len(at.exception) == 0


def test_ui_tab1_supervisor_matcher():
    """Verify Tab 1 Supervisor & Lab Matcher execution."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert len(at.exception) == 0

    # Submit match form
    at.button[0].click().run()
    assert len(at.exception) == 0


def test_ui_tab2_literature_gaps():
    """Verify Tab 2 Literature Gap Analyzer execution."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert len(at.exception) == 0

    # Submit gap analyzer form
    at.button[1].click().run()
    assert len(at.exception) == 0


def test_ui_tab3_coauthor_radar():
    """Verify Tab 3 Cross-Disciplinary Co-Author Radar execution."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert len(at.exception) == 0

    # Submit coauthor radar form
    at.button[2].click().run()
    assert len(at.exception) == 0


def test_ui_tab4_claim_audit():
    """Verify Tab 4 Verbatim Claim Evidence Matrix execution."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert len(at.exception) == 0

    # Submit claim audit form
    at.button[3].click().run()
    assert len(at.exception) == 0


def test_ui_tab5_academic_search():
    """Verify Tab 5 Academic Search execution."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert len(at.exception) == 0

    # Submit academic search form
    at.button[4].click().run()
    assert len(at.exception) == 0


def test_ui_tab6_diagnostics_benchmark():
    """Verify Tab 6 System Diagnostics execution."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert len(at.exception) == 0

    # Submit benchmark button
    at.button[5].click().run()
    assert len(at.exception) == 0
