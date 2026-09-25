"""Tests for the persistent memory store."""

from __future__ import annotations

from datetime import date

import pytest
from tests.test_worktree import git_sync, make_repo_sync

from lecode.config.models import Config
from lecode.memory.store import (
    MAX_SEARCH_HITS,
    MemoryStore,
    memory_injection,
    memory_root,
    project_slug,
)


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path / "mem")


# -- slug / root ---------------------------------------------------------------


def test_project_slug_stable_and_hashed(tmp_path):
    slug = project_slug(tmp_path)
    assert slug == project_slug(tmp_path)
    assert slug != project_slug(tmp_path / "other")
    # tail components + short hash suffix
    assert slug.rsplit("-", 1)[1].isalnum()
    assert len(slug.rsplit("-", 1)[1]) == 8


def test_memory_root_honors_config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    root = memory_root(tmp_path / "proj")
    assert root.parent == tmp_path / "cfg" / "memory"
    assert root.name == project_slug(tmp_path / "proj")


# -- long-term memory ------------------------------------------------------------


def test_project_root_shared_by_checkout_nested_cwd_and_linked_worktree(tmp_path):
    from lecode.memory.store import resolve_project_root

    repo = make_repo_sync(tmp_path / "main checkout")
    nested = repo / "src" / "nested"
    nested.mkdir(parents=True)
    linked = tmp_path / "linked checkout"
    git_sync(repo, "worktree", "add", "-b", "feature", str(linked))
    (linked / "nested").mkdir()
    for cwd in (repo, nested, linked, linked / "nested"):
        assert resolve_project_root(cwd) == repo
        assert resolve_project_root(resolve_project_root(cwd)) == repo
    assert resolve_project_root(repo / ".git") == repo
    outside = tmp_path / "outside"
    outside.mkdir()
    assert resolve_project_root(outside / ".." / "outside") == outside


def test_submodule_and_relative_git_metadata_keep_project_identity(tmp_path):
    import os

    from lecode.memory.store import resolve_project_root

    parent = make_repo_sync(tmp_path / "parent")
    source = make_repo_sync(tmp_path / "source")
    git_sync(parent, "-c", "protocol.file.allow=always", "submodule", "add", str(source), "sub")
    sub = parent / "sub"
    (sub / "nested").mkdir()
    linked = tmp_path / "sub linked"
    git_sync(sub, "worktree", "add", "-b", "feature", str(linked))
    (linked / "nested").mkdir()
    identity = resolve_project_root(sub)
    assert identity != resolve_project_root(parent)
    for cwd in (sub, sub / "nested", linked, linked / "nested"):
        assert resolve_project_root(cwd) == identity
        assert resolve_project_root(resolve_project_root(cwd)) == identity

    main_linked = tmp_path / "main linked"
    git_sync(parent, "worktree", "add", "-b", "feature", str(main_linked))
    marker = main_linked / ".git"
    gitdir = marker.read_text().strip().removeprefix("gitdir: ")
    marker.write_text(f"gitdir: {os.path.relpath(gitdir, main_linked)}\n")
    assert resolve_project_root(main_linked) == parent


def test_separate_gitdir_shares_identity_without_running_git(tmp_path, monkeypatch):
    import os
    import subprocess

    from lecode.memory.store import resolve_project_root

    repo = make_repo_sync(tmp_path / "checkout")
    git_sync(repo, "init", "--separate-git-dir", str(tmp_path / "git storage"))
    linked = tmp_path / "linked"
    git_sync(repo, "worktree", "add", "-b", "feature", str(linked))
    for checkout in (repo, linked):
        (checkout / "nested").mkdir()
        marker = checkout / ".git"
        gitdir = marker.read_text().strip().removeprefix("gitdir: ")
        marker.write_text(f"gitdir: {os.path.relpath(gitdir, checkout)}\n")

    def forbid_process(*args, **kwargs):
        raise AssertionError("project identity must use metadata, not a Git process")

    monkeypatch.setattr(subprocess, "run", forbid_process)
    identity = resolve_project_root(repo)
    for cwd in (repo, repo / "nested", linked, linked / "nested"):
        assert resolve_project_root(cwd) == identity
        assert resolve_project_root(resolve_project_root(cwd)) == identity


def test_legacy_migration_copies_markdown_only_once_without_merging(tmp_path):
    from lecode.memory.facts import FactStore
    from lecode.memory.store import migrate_legacy_memory

    repo = make_repo_sync(tmp_path / "repo")
    nested = repo / "nested"
    nested.mkdir()
    cfg = tmp_path / "cfg"
    legacy = MemoryStore(memory_root(nested, cfg))
    legacy.write_long_term("legacy")
    legacy.write_note("note", "keep")
    legacy.append_daily("daily", "2026-09-15")
    legacy.write_scratchpad("scratch")
    facts = FactStore(legacy.root / "facts.sqlite3")
    facts.add("do not copy sqlite", source_id="s", source_seq=0)
    facts.close()
    canonical = MemoryStore(memory_root(repo, cfg))

    migrate_legacy_memory(nested, repo, config_dir=cfg)

    assert canonical.read_long_term() == "legacy\n"
    assert canonical.read_note("note") == "keep\n"
    assert "daily" in canonical.read_daily("2026-09-15")
    assert canonical.read_scratchpad() == "scratch\n"
    assert not (canonical.root / "facts.sqlite3").exists()
    assert legacy.read_long_term() == "legacy\n"
    canonical.write_long_term("canonical wins")
    canonical.delete_note("note")
    migrate_legacy_memory(nested, repo, config_dir=cfg)
    assert canonical.read_long_term() == "canonical wins\n"
    assert canonical.read_note("note") is None


def test_migration_preserves_empty_destination_and_cleans_staging_on_error(tmp_path, monkeypatch):
    import shutil

    from lecode.memory.store import migrate_legacy_memory

    cfg = tmp_path / "cfg"
    cwd, project = tmp_path / "nested", tmp_path / "project"
    legacy = MemoryStore(memory_root(cwd, cfg))
    legacy.write_long_term("legacy")
    destination = memory_root(project, cfg)
    destination.mkdir()
    migrate_legacy_memory(cwd, project, config_dir=cfg)
    assert list(destination.iterdir()) == []
    destination.rmdir()

    def fail_copy(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(shutil, "copy2", fail_copy)
    with pytest.raises(OSError, match="disk full"):
        migrate_legacy_memory(cwd, project, config_dir=cfg)
    assert not destination.exists()
    assert list(destination.parent.iterdir()) == [legacy.root]
    assert legacy.read_long_term() == "legacy\n"


def test_long_term_round_trip(store):
    store.write_long_term("# Facts\n\nThe sky is blue.")
    assert store.read_long_term(capped=False) == "# Facts\n\nThe sky is blue.\n"


def test_long_term_missing_is_empty(store):
    assert store.read_long_term() == ""


def test_append_long_term_dated_headings(store):
    store.append_long_term("first entry")
    today = date.today().isoformat()
    content = store.read_long_term(capped=False)
    assert f"## {today}" in content
    assert "first entry" in content
    # second append the same day reuses the heading
    store.append_long_term("second entry")
    content = store.read_long_term(capped=False)
    assert content.count(f"## {today}") == 1
    assert "second entry" in content


def test_read_long_term_capped_with_marker(tmp_path):
    store = MemoryStore(tmp_path / "mem", max_bytes=50)
    store.write_long_term("x" * 200)
    content = store.read_long_term()
    assert "x" * 40 in content
    assert "memory truncated" in content
    assert len(content.encode()) < 100
    assert len(store.read_long_term(capped=False).encode()) == 201  # uncapped


def test_edit_long_term_replaces(store):
    store.write_long_term("alpha beta gamma")
    store.edit_long_term("beta", "BETA")
    assert "alpha BETA gamma" in store.read_long_term(capped=False)


def test_edit_long_term_not_found(store):
    store.write_long_term("alpha")
    with pytest.raises(KeyError):
        store.edit_long_term("missing", "x")


def test_edit_long_term_ambiguous(store):
    store.write_long_term("dup and dup")
    with pytest.raises(ValueError, match="ambiguous"):
        store.edit_long_term("dup", "x")


# -- daily logs --------------------------------------------------------------------


def test_daily_log_round_trip(store):
    store.append_daily("did a thing")
    content = store.read_daily()
    assert "# Daily log" in content
    assert "did a thing" in content
    assert "##" in content  # timestamped heading


def test_daily_log_explicit_date(store):
    store.append_daily("old entry", day="2020-01-02")
    assert "old entry" in store.read_daily("2020-01-02")
    assert store.read_daily("1999-12-31") == ""


def test_write_daily_overwrites(store):
    store.append_daily("first")
    store.write_daily("replacement")
    content = store.read_daily()
    assert "replacement" in content
    assert "first" not in content


def test_flush_summary_under_compaction(store):
    store.append_daily("morning work")
    store.flush_summary("session so far: explored auth")
    content = store.read_daily()
    assert "## Compaction" in content
    assert "session so far: explored auth" in content
    assert "morning work" in content


# -- scratchpad ---------------------------------------------------------------------


def test_scratchpad_round_trip(store):
    store.write_scratchpad("- [ ] task one\n- [x] task two")
    assert store.read_scratchpad() == "- [ ] task one\n- [x] task two\n"


def test_scratchpad_missing_is_empty(store):
    assert store.read_scratchpad() == ""


# -- notes ---------------------------------------------------------------------------


def test_notes_crud(store):
    store.write_note("todo", "buy milk")
    assert store.read_note("todo") == "buy milk\n"
    assert store.list_notes() == ["todo"]
    store.write_note("ideas", "a, b, c")
    assert store.list_notes() == ["ideas", "todo"]
    assert store.delete_note("todo") is True
    assert store.read_note("todo") is None
    assert store.delete_note("todo") is False


def test_note_name_validated(store):
    with pytest.raises(ValueError):
        store.write_note("../evil", "nope")
    with pytest.raises(ValueError):
        store.read_note("BAD NAME")


# -- atomic writes + backups ----------------------------------------------------------


def test_bak_created_on_overwrite(store):
    store.write_long_term("version one")
    assert not (store.long_term_path.parent / "MEMORY.md.bak").exists()
    store.write_long_term("version two")
    bak = store.long_term_path.parent / "MEMORY.md.bak"
    assert bak.read_text(encoding="utf-8") == "version one\n"


def test_no_tmp_files_left(store):
    store.write_long_term("a")
    store.append_daily("b")
    store.write_note("n", "c")
    store.flush_summary("d")
    tmp_files = list(store.root.rglob("*.tmp"))
    assert tmp_files == []


# -- search ---------------------------------------------------------------------------


def test_search_ranking_and_fields(store):
    store.write_note("zzz", "keyword in a note")
    store.append_daily("keyword in daily", day="2020-01-01")
    store.append_daily("keyword in recent daily", day="2020-06-01")
    store.write_long_term("keyword in MEMORY")
    hits = store.search("keyword")
    files = [h.file for h in hits]
    assert files[0] == "MEMORY.md"
    # daily logs recent-first, before notes
    assert files.index("daily/2020-06-01.md") < files.index("daily/2020-01-01.md")
    assert files.index("daily/2020-01-01.md") < files.index("notes/zzz.md")
    hit = hits[0]
    assert hit.line_no >= 1
    assert "keyword" in hit.line
    assert "keyword" in hit.snippet


def test_search_case_insensitive(store):
    store.write_long_term("The Keyword Is Here")
    assert store.search("keyword")
    assert store.search("KEYWORD")


def test_search_invalid_regex(store):
    with pytest.raises(ValueError, match="invalid regex"):
        store.search("[unclosed")


def test_search_cap(store):
    store.write_long_term("\n".join(f"hit line {i}" for i in range(MAX_SEARCH_HITS + 20)))
    hits = store.search("hit line")
    assert len(hits) == MAX_SEARCH_HITS


def test_search_no_matches(store):
    store.write_long_term("nothing here")
    assert store.search("absent") == []


def test_search_scratchpad_included(store):
    store.write_scratchpad("- [ ] keyword task")
    hits = store.search("keyword")
    assert hits[0].file == "scratchpad.md"


# -- injection -------------------------------------------------------------------------


def test_injection_absent_when_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    assert memory_injection(Config(), tmp_path) is None


def test_injection_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    store = MemoryStore(memory_root(tmp_path))
    store.write_long_term("remember this")
    config = Config()
    config.memory.enabled = False
    assert memory_injection(config, tmp_path) is None


def test_injection_content(tmp_path, monkeypatch):
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    store = MemoryStore(memory_root(tmp_path))
    store.write_long_term("remember this")
    store.write_scratchpad("- [ ] pending item")
    section = memory_injection(Config(), tmp_path)
    assert section.startswith("## Memory")
    assert "### Long-term memory" in section
    assert "remember this" in section
    assert "### Scratchpad" in section
    assert "pending item" in section


def test_injection_cap_applies(tmp_path, monkeypatch):
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    store = MemoryStore(memory_root(tmp_path))
    store.write_long_term("y" * 40000)
    section = memory_injection(Config(), tmp_path)
    assert "memory truncated" in section
    assert len(section.encode()) < 33000 + 200
