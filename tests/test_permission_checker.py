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


def _checker(permissions: dict, session_perms=None, cwd=None, read_only=False) -> PermissionChecker:
    return PermissionChecker(
        Config.model_validate({"permissions": permissions}),
        session_perms,
        cwd=cwd,
        read_only=read_only,
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
        ("ask_user", {"questions": []}, Decision.ALLOW),
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


# -- read-only enforcement ------------------------------------------------------


def test_read_only_denies_writes_even_in_yolo():
    checker = _checker({"mode": "yolo"}, read_only=True)
    assert checker.read_only is True
    assert checker.check("bash", {"command": "ls"}).decision == Decision.DENY
    assert checker.check("write", {"path": "a.py"}).decision == Decision.DENY
    assert checker.check("edit", {"path": "a.py"}).decision == Decision.DENY
    assert "read-only" in checker.check("bash", {"command": "ls"}).reason


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({"action": "question", "text": "Need a choice"}, Decision.ALLOW),
        ({"action": "list"}, Decision.ALLOW),
        ({"action": "inspect", "id": "w"}, Decision.ALLOW),
        ({"action": "send", "id": "w", "text": "continue"}, Decision.ALLOW),
        ({"action": "stop", "id": "w"}, Decision.DENY),
        ({"action": "resume", "id": "w"}, Decision.DENY),
        ({"action": "submit", "id": "w"}, Decision.DENY),
        ({"action": "integrate"}, Decision.DENY),
        ({"action": "cleanup"}, Decision.DENY),
        ({"action": "recover"}, Decision.DENY),
        ({"action": "unknown"}, Decision.DENY),
        ({}, Decision.DENY),
        ({"action": None}, Decision.DENY),
        ({"action": 1}, Decision.DENY),
        ({"action": True}, Decision.DENY),
        ({"action": ["send"]}, Decision.DENY),
        ({"action": {"send": True}}, Decision.DENY),
    ],
)
def test_readonly_workers_control_plane(args, expected, strict):
    checker = _checker({"mode": "yolo" if strict else "readonly"}, read_only=strict)
    assert checker.check("workers", args).decision == expected


@pytest.mark.parametrize("action", ["list", "question", "send", "inspect"])
@pytest.mark.parametrize("decision", [Decision.ASK, Decision.DENY])
@pytest.mark.parametrize("source", ["global", "overlay"])
def test_readonly_worker_controls_honor_rules_through_descendants(action, decision, source):
    rules = {decision: {"workers": [{"pattern": "*"}]}}
    parent = _checker(
        {"mode": "yolo", "rules": rules if source == "global" else {}}, read_only=True
    ).for_agent(AgentOverlay(extra_rules=_ruleset(**rules) if source == "overlay" else _ruleset()))
    child = parent.for_child(
        AgentOverlay(extra_rules=_ruleset(allow={"workers": [{"pattern": "*"}]})),
        session_perms=SessionPermissions([("workers", "*")]),
    ).for_child(AgentOverlay(mode="yolo"))
    result = child.check("workers", {"action": action, "id": "w", "text": "reply"})
    assert result.decision == decision
    assert result.matched_rule == PermissionRule(pattern="*")


@pytest.mark.parametrize("action", ["list", "question", "send", "inspect"])
def test_readonly_worker_controls_honor_denied_tools(action):
    checker = _checker({"mode": "readonly"}).for_agent(AgentOverlay(denied_tools=("workers",)))
    child = checker.for_child(AgentOverlay(mode="yolo"))
    assert child.check("workers", {"action": action}).decision == Decision.DENY


def test_readonly_parent_reply_does_not_widen_child_permissions():
    parent = _checker({"mode": "yolo"}, read_only=True)
    child = parent.for_child(
        AgentOverlay(
            mode="yolo",
            extra_rules=_ruleset(
                allow={"write": [{"pattern": "*"}], "workers": [{"pattern": "*"}]}
            ),
        ),
        session_perms=SessionPermissions([("write", "*"), ("workers", "*")]),
    )
    assert parent.check("workers", {"action": "send", "id": "w", "text": "write"}).decision == (
        Decision.ALLOW
    )
    assert child.read_only is True
    assert child.check("write", {"path": "a.py"}).decision == Decision.DENY
    assert child.check("workers", {"action": "integrate"}).decision == Decision.DENY


def test_read_only_not_widened_by_overlay_allow_rule():
    overlay = AgentOverlay(extra_rules=_ruleset(allow={"bash": [{"pattern": "*"}]}))
    checker = _checker({"mode": "yolo"}, read_only=True).for_agent(overlay)
    assert checker.check("bash", {"command": "ls"}).decision == Decision.DENY


def test_read_only_not_widened_by_session_grant():
    perms = SessionPermissions([("bash", "*")])
    checker = _checker({"mode": "yolo"}, session_perms=perms, read_only=True)
    assert checker.check("bash", {"command": "ls"}).decision == Decision.DENY


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("read", {"path": "a.py"}),
        ("grep", {"pattern": "x"}),
        ("list_dir", {"path": "."}),
        ("task", {"prompt": "inspect"}),
    ],
)
def test_read_only_allows_read_class_tools(tool, args):
    checker = _checker({"mode": "yolo"}, read_only=True)
    assert checker.check(tool, args).decision == Decision.ALLOW


def test_for_agent_propagates_read_only():
    checker = _checker({"mode": "yolo"}, read_only=True)
    derived = checker.for_agent(BUILTIN_AGENT_OVERLAYS["build"])
    assert derived.read_only is True
    assert derived.check("write", {"path": "a"}).decision == Decision.DENY


def test_for_agent_read_only_narrows_writable_checker():
    checker = _checker({"mode": "yolo"})
    derived = checker.for_agent(BUILTIN_AGENT_OVERLAYS["build"], read_only=True)
    assert derived.read_only is True
    assert checker.read_only is False
    assert checker.check("write", {"path": "a"}).decision == Decision.ALLOW
    assert derived.check("write", {"path": "a"}).decision == Decision.DENY


def test_for_agent_cannot_widen_read_only_checker():
    checker = _checker({"mode": "yolo"}, read_only=True)
    derived = checker.for_agent(BUILTIN_AGENT_OVERLAYS["build"], read_only=False)
    assert derived.read_only is True
    assert derived.check("bash", {"command": "ls"}).decision == Decision.DENY


def test_nested_agent_overlays_retain_each_ancestor_restriction():
    parent = _checker(
        {"mode": "yolo", "rules": {"ask": {"read": [{"pattern": "global/*"}]}}}
    ).for_agent(
        AgentOverlay(
            denied_tools=("bash",),
            extra_rules=_ruleset(
                deny={"read": [{"pattern": "denied/*"}]},
                ask={"read": [{"pattern": "private/*"}]},
            ),
        )
    )
    child = parent.for_agent(
        AgentOverlay(extra_rules=_ruleset(allow={"read": [{"pattern": "*"}]}))
    ).for_agent(AgentOverlay(mode="yolo"))
    for path, expected in (
        ("global/a", Decision.ASK),
        ("denied/a", Decision.DENY),
        ("private/a", Decision.ASK),
        ("public/a", Decision.ALLOW),
    ):
        assert child.check("read", {"path": path}).decision == expected
    assert child.check("bash", {"command": "ls"}).decision == Decision.DENY


def test_child_rules_and_grants_cannot_override_ancestor_ask_or_deny():
    parent = _checker({"mode": "yolo"}).for_agent(
        AgentOverlay(
            denied_tools=("write",),
            extra_rules=_ruleset(
                ask={"read": [{"pattern": "private/*"}]},
                deny={"read": [{"pattern": "denied/*"}]},
            ),
        )
    )
    grants = SessionPermissions([("read", "*"), ("write", "*")])
    child = parent.for_child(
        AgentOverlay(extra_rules=_ruleset(allow={"read": [{"pattern": "*"}]})),
        session_perms=grants,
    ).for_child(AgentOverlay(mode="yolo"))
    assert child.check("read", {"path": "private/a"}).decision == Decision.ASK
    assert child.check("read", {"path": "denied/a"}).decision == Decision.DENY
    assert child.check("write", {"path": "a"}).decision == Decision.DENY
    assert child.check("read", {"path": "public/a"}).decision == Decision.ALLOW


@pytest.mark.parametrize("derive", ["for_agent", "for_child"])
def test_readonly_overlay_remains_effective_through_writable_descendants(derive):
    parent = _checker({"mode": "yolo"}).for_agent(AgentOverlay(mode="readonly"))
    child = getattr(parent, derive)(
        AgentOverlay(mode="yolo", extra_rules=_ruleset(allow={"write": [{"pattern": "*"}]}))
    )
    child.set_mode("yolo")
    assert parent.read_only is True
    assert child.read_only is True
    assert child.mode == "readonly"
    assert child.check("write", {"path": "a"}).decision == Decision.DENY
    assert child.check("read", {"path": "a"}).decision == Decision.ALLOW


def test_child_path_rules_use_child_cwd_for_all_ancestor_layers(tmp_path):
    parent_cwd, child_cwd = tmp_path / "parent", tmp_path / "child"
    parent = _checker(
        {
            "mode": "yolo",
            "rules": {"deny": {"read": [{"pattern": str(child_cwd / "secret")}]}},
        },
        cwd=parent_cwd,
    ).for_agent(
        AgentOverlay(
            extra_rules=_ruleset(ask={"read": [{"pattern": str(child_cwd / "review")}]}),
        )
    )
    child = parent.for_child(cwd=child_cwd).for_child()
    assert child.check("read", {"path": "secret"}).decision == Decision.DENY
    assert child.check("read", {"path": "review"}).decision == Decision.ASK
    assert parent.check("read", {"path": "secret"}).decision == Decision.ALLOW
    assert parent.check("read", {"path": "review"}).decision == Decision.ALLOW
    assert child.check("read", {"path": str(parent_cwd / "secret")}).decision == Decision.ALLOW


def test_children_have_independent_doom_tracking_without_recording_parent_calls():
    parent = _checker({"mode": "yolo"}).for_agent(AgentOverlay())
    args = {"path": "same"}
    assert parent.check("read", args).decision == Decision.ALLOW
    assert parent.check("read", args).decision == Decision.ALLOW
    child, sibling = parent.for_child(), parent.for_child()
    grandchild = child.for_child()
    for checker in (child, sibling, grandchild):
        assert [checker.check("read", args).decision for _ in range(4)] == [
            Decision.ALLOW,
            Decision.ALLOW,
            Decision.ASK,
            Decision.DENY,
        ]
    assert parent.check("read", args).decision == Decision.ASK


def test_child_uses_supplied_scoped_grants_without_sharing_parent_or_sibling_grants():
    parent_grants = SessionPermissions([("write", "src/*")])
    parent = _checker({"mode": "readonly"}, session_perms=parent_grants)
    child_grants = SessionPermissions()
    child = parent.for_child(session_perms=child_grants)
    sibling = parent.for_child()
    assert child.check("write", {"path": "src/before"}).decision == Decision.DENY
    child_grants.grant("write", "src/approved")
    child_grants.grant("write", "outside/*")
    assert child.check("write", {"path": "src/approved"}).decision == Decision.ALLOW
    assert child.check("write", {"path": "src/other"}).decision == Decision.DENY
    assert child.check("write", {"path": "outside/file"}).decision == Decision.DENY
    assert sibling.check("write", {"path": "src/approved"}).decision == Decision.DENY
    assert parent.check("write", {"path": "src/other"}).decision == Decision.ALLOW
    assert parent.check("write", {"path": "outside/file"}).decision == Decision.DENY
    assert parent_grants.grants == [("write", "src/*")]


@pytest.mark.parametrize("source", ["global_allow", "global_ask", "grant", "overlay_allow"])
def test_readonly_with_writable_exceptions_is_not_safe_for_shared_checkout(source):
    permissions = {"mode": "readonly"}
    grants = SessionPermissions()
    overlay = AgentOverlay()
    expected = Decision.ALLOW
    if source.startswith("global_"):
        decision = source.removeprefix("global_")
        permissions["rules"] = {decision: {"write": [{"pattern": "allowed/*"}]}}
        expected = Decision(decision)
    elif source == "grant":
        grants.grant("write", "allowed/*")
    else:
        permissions["mode"] = "yolo"
        overlay = AgentOverlay(
            mode="readonly",
            extra_rules=_ruleset(allow={"write": [{"pattern": "allowed/*"}]}),
        )
    checker = _checker(permissions, session_perms=grants).for_agent(overlay)
    assert checker.read_only is False
    assert checker.check("write", {"path": "allowed/a"}).decision == expected
    assert checker.check("write", {"path": "other/a"}).decision == Decision.DENY
    strict = checker.for_child(read_only=True)
    assert strict.read_only is True
    assert strict.check("write", {"path": "allowed/a"}).decision == Decision.DENY


@pytest.mark.parametrize("source", ["global", "overlay", "grant"])
@pytest.mark.parametrize("decision", [Decision.ALLOW, Decision.ASK])
def test_readonly_workers_exceptions_are_not_safe_for_shared_checkout(source, decision):
    rules = {decision: {"workers": [{"pattern": "*"}]}}
    checker = _checker(
        {
            "mode": "yolo" if source == "overlay" else "readonly",
            "rules": rules if source == "global" else {},
        },
        session_perms=SessionPermissions([("workers", "*")] if source == "grant" else []),
    ).for_agent(
        AgentOverlay(
            mode="readonly", extra_rules=_ruleset(**rules) if source == "overlay" else _ruleset()
        )
    )
    assert checker.read_only is False
    assert checker.check("workers", {"action": "integrate"}).decision == (
        Decision.ALLOW if source == "grant" else decision
    )
    strict = checker.for_child(read_only=True)
    assert strict.read_only is True
    assert strict.check("workers", {"action": "integrate"}).decision == Decision.DENY


def test_child_readonly_capability_uses_base_policy_despite_yolo_overlay():
    parent = _checker({"mode": "readonly"}, session_perms=SessionPermissions([("write", "*")]))
    child = parent.for_child(AgentOverlay(mode="yolo"))
    assert parent.read_only is False
    assert child.read_only is True
    assert child.check("write", {"path": "a.py"}).decision == Decision.DENY
    assert child.check("workers", {"action": "send", "id": "w", "text": "reply"}).decision == (
        Decision.ALLOW
    )


@pytest.mark.parametrize("parent_decision", list(Decision))
@pytest.mark.parametrize("overlay_decision", list(Decision))
def test_per_call_overlay_composes_with_full_policy(parent_decision, overlay_decision):
    parent = _checker(
        {"mode": "yolo", "rules": {parent_decision: {"read": [{"pattern": "*"}]}}}
    ).for_agent(AgentOverlay(denied_tools=("write",)))
    overlay = AgentOverlay(extra_rules=_ruleset(**{overlay_decision: {"read": [{"pattern": "*"}]}}))
    if Decision.DENY in (parent_decision, overlay_decision):
        expected = Decision.DENY
    elif Decision.ASK in (parent_decision, overlay_decision):
        expected = Decision.ASK
    else:
        expected = Decision.ALLOW
    assert parent.check("read", {"path": "a"}, overlay).decision == expected
    assert parent.check("write", {"path": "a"}, overlay).decision == Decision.DENY
    assert parent.check("read", {"path": "b"}).decision == parent_decision
