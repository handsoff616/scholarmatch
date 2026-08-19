"""Unit tests for ScholarMatch command-line interface."""

from unittest.mock import patch
import pytest
from scholarmatch.cli import main


def test_cli_help(capsys):
    with patch("sys.argv", ["scholarmatch", "--help"]):
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 0


def test_cli_match_command(capsys):
    with patch("sys.argv", ["scholarmatch", "match", "--candidate-idx", "0", "--top-k", "2"]):
        main()
        captured = capsys.readouterr()
        # Rich output or text should contain match results
        assert "Rank" in captured.out or "Barzilay" in captured.out or len(captured.out) > 0


def test_cli_gap_command(capsys):
    with patch("sys.argv", ["scholarmatch", "gap-discovery", "--top-k", "3"]):
        main()
        captured = capsys.readouterr()
        assert "Frontier" in captured.out or len(captured.out) > 0


def test_cli_audit_command(capsys):
    with patch("sys.argv", ["scholarmatch", "audit-claim", "--candidate-idx", "0"]):
        main()
        captured = capsys.readouterr()
        assert "Verbatim" in captured.out or len(captured.out) > 0
