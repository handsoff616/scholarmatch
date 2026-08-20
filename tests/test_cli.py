"""Unit tests for ScholarMatch command-line interface."""

from unittest.mock import patch
import pytest
from scholarmatch.cli import main


def test_cli_help(capsys):
    with patch("sys.argv", ["scholarmatch", "--help"]):
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 0
        captured = capsys.readouterr()
        assert "ScholarMatch" in captured.out
        assert "match" in captured.out
        assert "gap-discovery" in captured.out
        assert "audit-claim" in captured.out


def test_cli_match_command(capsys):
    with patch("sys.argv", ["scholarmatch", "match", "--candidate-idx", "0", "--top-k", "2"]):
        main()
        captured = capsys.readouterr()
        assert len(captured.out) > 0
        assert "Faculty" in captured.out or "Rank" in captured.out or "Barzilay" in captured.out


def test_cli_gap_command(capsys):
    with patch("sys.argv", ["scholarmatch", "gap-discovery", "--top-k", "3"]):
        main()
        captured = capsys.readouterr()
        assert "Frontier" in captured.out or "Rank" in captured.out


def test_cli_audit_command(capsys):
    with patch("sys.argv", ["scholarmatch", "audit-claim", "--candidate-idx", "0"]):
        main()
        captured = capsys.readouterr()
        assert "Evidence" in captured.out or "Verbatim" in captured.out


def test_cli_benchmark_command(capsys):
    with patch("sys.argv", ["scholarmatch", "benchmark"]):
        main()
        captured = capsys.readouterr()
        assert "Benchmark" in captured.out or "Latency" in captured.out
