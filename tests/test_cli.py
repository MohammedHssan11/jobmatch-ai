import json
import pytest
from pathlib import Path
from cli import build_parser, read_text_or_file, cmd_anonymize, cmd_bias_check


def test_build_parser_subcommands():
    parser = build_parser()
    # Test 'match' args
    args_match = parser.parse_args(["match", "--resume", "dummy_cv.txt", "--job", "dummy_job.txt", "--json"])
    assert args_match.command == "match"
    assert args_match.json is True

    # Test 'rank-jobs' args
    args_rank = parser.parse_args(["rank-jobs", "--resume", "dummy_cv.txt", "--top-k", "10", "--cross-attention"])
    assert args_rank.command == "rank-jobs"
    assert args_rank.top_k == 10
    assert args_rank.cross_attention is True

    # Test 'heatmap' args
    args_hm = parser.parse_args(["heatmap", "--resume", "dummy_cv.txt", "--job", "dummy_job.txt", "--export-svg", "out.svg"])
    assert args_hm.command == "heatmap"
    assert args_hm.export_svg == "out.svg"

    # Test 'anonymize' args
    args_anon = parser.parse_args(["anonymize", "--resume", "John Doe 555-1234"])
    assert args_anon.command == "anonymize"

    # Test 'bias-check' args
    args_bias = parser.parse_args(["bias-check", "--job", "Rockstar developer"])
    assert args_bias.command == "bias-check"


def test_read_text_or_file_raw_string():
    raw = "Raw text input"
    assert read_text_or_file(raw) == raw


def test_read_text_or_file_from_file(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("File content here", encoding="utf-8")
    assert read_text_or_file(str(f)) == "File content here"


def test_cmd_anonymize_to_file(tmp_path, capsys):
    out_file = tmp_path / "anon.txt"
    parser = build_parser()
    args = parser.parse_args(["anonymize", "--resume", "Contact Alice at alice@example.com", "--output", str(out_file)])
    cmd_anonymize(args)

    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "alice@example.com" not in content
    assert "[REDACTED EMAIL]" in content


def test_cmd_bias_check(capsys):
    parser = build_parser()
    args = parser.parse_args(["bias-check", "--job", "We need a rockstar ninja developer who is aggressive."])
    cmd_bias_check(args)

    captured = capsys.readouterr()
    assert "INCLUSIVITY" in captured.out
    assert "ninja" in captured.out
