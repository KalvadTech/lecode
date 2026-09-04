"""Tests for the permission checker: pipeline ordering, modes, overlays, doom loop."""

from __future__ import annotations

import pytest

from lecode.config.models import Config, PermissionRule, PermissionRuleSet
from lecode.permission import (
    BUILTIN_AGENT_OVERLAYS,
    AgentOverlay,
    Decision,
    PermissionChecker,
    SessionPermissions,
)


def _checker(permissions: dict, session_perms=None, cwd=None) -> PermissionChecker:
    return PermissionChecker(
        Config.model_validate({"permissions": permissions}), session_perms, cwd=cwd
    )


def _ruleset(**tables) -> PermissionRuleSet:
    """ruleset(allow={...}, ask={...}, deny={...}) with compact rule values."""
    return PermissionRuleSet.model_validate(tables)


# -- ordering: rules beat mode fallback --------------------------------------


def test_allow_rule_overrides_readonly_deny():
    checker = _checker({"mode": "readonly", "rules": {"allow": {"bash": [{"pattern": "git *"}]}}})
    result = checker.check("bash", {"command": "git status"})
    assert result.decision == Decision.ALLOW
    assert result.matched_rule == PermissionRule(pattern="git *")


def test_ask_rule_overrides_yolo_allow():
    checker = _checker({"mode": "yolo", "rules": {"ask": {"read": [{"pattern": "*.env"}]}}})
    result = checker.check("read", {"path": "app.env"})
    assert result.decision == Decision.ASK


def test_last_match_wins_within_tool():
    checker = _checker(
        {
            "mode": "yolo",
            "rules": {
                "allow": {
                    "bash": [{"pattern": "git *"}, {"pattern": "git push *"}],
                },
                "ask": {"bash": [{"pattern": "git push *"}]},
            },
        }
    )
    # ask rule walked after allow rules -> last match wins
    assert checker.check("bash", {"command": "git push origin"}).decision == Decision.ASK
    assert checker.check("bash", {"command": "git status"}).decision == Decision.ALLOW


def test_deny_is_unbypassable():
    checker = _checker(
        {
            "mode": "yolo",
            "rules": {
                "deny": {"bash": [{"pattern": "rm -rf *"}]},
                "allow": {"bash": [{"pattern": "*"}]},
            },
        },
        session_perms=SessionPermissions([("bash", "rm -rf *")]),
    )
    result = checker.check("bash", {"command": "rm -rf build/"})
    assert result.decision == Decision.DENY
    assert result.matched_rule is not None


def test_deny_wins_regardless_of_position():
    checker = _checker(
        {
            "mode": "yolo",
            "rules": {
                "allow": {"bash": [{"pattern": "rm *"}]},
                "deny": {"bash": [{"pattern": "rm -rf *"}]},
            },
        }
    )
    assert checker.check("bash", {"command": "rm -rf x"}).decision == Decision.DENY


def test_regex_deny_rule():
    checker = _checker(
        {"rules": {"deny": {"bash": [{"pattern": r"\bgit\s+push\b", "kind": "regex"}]}}}
    )
    assert checker.check("bash", {"command": "git push origin main"}).decision == Decision.DENY


# -- session allowlist --------------------------------------------------------


def test_session_grant_allows_after_rules():
    perms = SessionPermissions([("bash", "git status*")])
    checker = _checker({"mode": "readonly"}, session_perms=perms)
    result = checker.check("bash", {"command": "git status --short"})
    assert result.decision == Decision.ALLOW
    assert "grant" in result.reason


def test_session_grant_does_not_cover_other_commands():
    perms = SessionPermissions([("bash", "git status*")])
    checker = _checker({"mode": "readonly"}, session_perms=perms)
    assert checker.check("bash", {"command": "git push"}).decision == Decision.DENY


def test_grant_method_appends():
    perms = SessionPermissions()
    perms.grant("write", "src/**")
    checker = _checker({"mode": "readonly"}, session_perms=perms)
    assert checker.check("write", {"path": "src/a.py"}).decision == Decision.ALLOW


# -- two modes -----------------------------------------------------------------


def test_yolo_allows_everything():
    checker = _checker({"mode": "yolo"})
    assert checker.check("bash", {"command": "rm -rf build"}).decision == Decision.ALLOW
    assert checker.check("write", {"path": "x.py"}).decision == Decision.ALLOW
    assert checker.check("mystery_tool", {}).decision == Decision.ALLOW


@pytest.mark.parametrize(
    ("tool", "args", "expected"),
    [
        ("read", {"path": "a.py"}, Decision.ALLOW),
        ("grep", {"pattern": "x"}, Decision.ALLOW),
        ("find_files", {"pattern": "*.py"}, Decision.ALLOW),
        ("list_dir", {"path": "."}, Decision.ALLOW),
        ("lsp_diagnostics", {"path": "a.py"}, Decision.ALLOW),
        ("memory_read", {}, Decision.ALLOW),
        ("memory_search", {"query": "x"}, Decision.ALLOW),
        ("write", {"path": "a.py"}, Decision.DENY),
        ("edit", {"path": "a.py"}, Decision.DENY),
        ("bash", {"command": "ls"}, Decision.DENY),
        ("todo_write", {}, Decision.DENY),
        ("memory_write", {}, Decision.DENY),
        ("unknown_tool", {}, Decision.DENY),
    ],
)
def test_readonly_mode_matrix(tool, args, expected):
    assert _checker({"mode": "readonly"}).check(tool, args).decision == expected


# -- MCP read-equivalence -------------------------------------------------------


@pytest.mark.parametrize("tool", ["mcp:exa:web_search", "mcp__context7__docs", "mcp:grep-app:s"])
def test_mcp_read_equiv_in_readonly(tool):
    assert _checker({"mode": "readonly"}).check(tool, {}).decision == Decision.ALLOW


def test_other_mcp_servers_not_read_equiv():
    checker = _checker({"mode": "readonly"})
    assert checker.check("mcp:filesystem:write_file", {}).decision == Decision.DENY


# -- doom loop --------------------------------------------------------------------


def test_doom_loop_third_call_asks():
    checker = _checker({"mode": "yolo"})
    args = {"command": "make test"}
    assert checker.check("bash", args).decision == Decision.ALLOW
    assert checker.check("bash", args).decision == Decision.ALLOW
    third = checker.check("bash", args)
    assert third.decision == Decision.ASK
    assert "3 times" in third.reason


def test_doom_loop_fourth_call_denies():
    checker = _checker({"mode": "yolo"})
    args = {"command": "make test"}
    for _ in range(3):
        checker.check("bash", args)
    fourth = checker.check("bash", args)
    assert fourth.decision == Decision.DENY
    assert "4 times" in fourth.reason


def test_doom_loop_non_consecutive_no_trigger():
    checker = _checker({"mode": "yolo"})
    a, b = {"command": "make"}, {"command": "ls"}
    checker.check("bash", a)
    checker.check("bash", b)
    checker.check("bash", a)
    checker.check("bash", b)
    assert checker.check("bash", a).decision == Decision.ALLOW  # 3rd, but not consecutive


def test_doom_loop_different_args_no_trigger():
    checker = _checker({"mode": "yolo"})
    for i in range(5):
        assert checker.check("bash", {"command": f"cmd{i}"}).decision == Decision.ALLOW


def test_doom_loop_deny_stays_deny():
    checker = _checker({"mode": "readonly"})
    args = {"path": "a.py"}
    for _ in range(4):
        assert checker.check("write", args).decision == Decision.DENY


# -- agent overlays -----------------------------------------------------------------


def test_plan_overlay_narrows_global_yolo():
    checker = _checker({"mode": "yolo"})
    plan = checker.for_agent(BUILTIN_AGENT_OVERLAYS["plan"])
    assert plan.check("read", {"path": "a"}).decision == Decision.ALLOW
    assert plan.check("write", {"path": "a"}).decision == Decision.DENY
    assert plan.check("bash", {"command": "ls"}).decision == Decision.DENY
    # the base checker is unaffected
    assert checker.check("write", {"path": "a"}).decision == Decision.ALLOW


def test_build_overlay_does_not_narrow():
    checker = _checker({"mode": "yolo"})
    build = checker.for_agent(BUILTIN_AGENT_OVERLAYS["build"])
    assert build.check("write", {"path": "a"}).decision == Decision.ALLOW


def test_overlay_denied_tools():
    overlay = AgentOverlay(denied_tools=("bash",))
    checker = _checker({"mode": "yolo"}).for_agent(overlay)
    assert checker.check("bash", {"command": "ls"}).decision == Decision.DENY
    assert checker.check("write", {"path": "a"}).decision == Decision.ALLOW


def test_overlay_extra_rules_evaluated_before_global():
    overlay = AgentOverlay(
        extra_rules=_ruleset(ask={"read": [{"pattern": "secrets/**"}]}),
    )
    checker = _checker(
        {"mode": "yolo", "rules": {"allow": {"read": [{"pattern": "**"}]}}}
    ).for_agent(overlay)
    result = checker.check("read", {"path": "secrets/key.pem"})
    assert result.decision == Decision.ASK
    # global allow rule still applies to non-overlay-matched targets
    assert checker.check("read", {"path": "src/a.py"}).decision == Decision.ALLOW


def test_overlay_deny_rules_unbypassable():
    overlay = AgentOverlay(extra_rules=_ruleset(deny={"read": [{"pattern": "*.pem"}]}))
    checker = _checker({"mode": "yolo"}).for_agent(overlay)
    assert checker.check("read", {"path": "key.pem"}).decision == Decision.DENY


def test_overlay_via_check_argument():
    checker = _checker({"mode": "yolo"})
    overlay = AgentOverlay(mode="readonly")
    result = checker.check("write", {"path": "a"}, agent_overlay=overlay)
    assert result.decision == Decision.DENY
    # subsequent plain check unaffected
    assert checker.check("write", {"path": "a"}).decision == Decision.ALLOW


def test_overlay_shares_doom_tracking():
    checker = _checker({"mode": "yolo"})
    plan = checker.for_agent(BUILTIN_AGENT_OVERLAYS["plan"])
    args = {"path": "a.py"}
    checker.check("read", args)
    plan.check("read", args)
    third = checker.check("read", args)
    assert third.decision == Decision.ASK  # count shared across derived checkers
