"""The ``index`` tool: a tree-sitter skeleton of a source file.

Lists imports, types, and function/method signatures with line ranges so the
agent can read only the ranges it needs instead of whole files. Unsupported
extensions produce a plain note, never an exception.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from lecode.agent.tools.base import Tool, ToolContext, ToolResult

if TYPE_CHECKING:
    import tree_sitter

#: Cap on emitted skeleton items (deep hierarchies stay readable).
MAX_ITEMS = 200
#: Cap on one signature line.
MAX_SIG_CHARS = 160


class _Spec:
    """Per-language extraction: definition node types and containers."""

    def __init__(
        self,
        loader: Any,
        defs: dict[str, str],
        containers: frozenset[str] = frozenset(),
    ) -> None:
        self.loader = loader  # () -> capsule for tree_sitter.Language
        self.defs = defs  # node type -> kind label
        self.containers = containers  # def types whose body nests deeper


def _ts(module_name: str, attr: str = "language") -> Any:
    def load() -> Any:
        import importlib

        return getattr(importlib.import_module(module_name), attr)()

    return load


_PY_DEFS = {
    "import_statement": "import",
    "import_from_statement": "import",
    "function_definition": "def",
    "class_definition": "class",
}
_JS_DEFS = {
    "import_statement": "import",
    "function_declaration": "function",
    "generator_function_declaration": "function",
    "class_declaration": "class",
    "method_definition": "method",
}
_TS_DEFS = _JS_DEFS | {
    "interface_declaration": "interface",
    "type_alias_declaration": "type",
    "enum_declaration": "enum",
    "abstract_class_declaration": "class",
    "internal_module": "namespace",
}
_C_DEFS = {
    "preproc_include": "include",
    "function_definition": "function",
    "type_definition": "typedef",
}

_SPECS: dict[str, _Spec] = {
    "python": _Spec(_ts("tree_sitter_python"), _PY_DEFS, frozenset({"class_definition"})),
    "javascript": _Spec(_ts("tree_sitter_javascript"), _JS_DEFS, frozenset({"class_declaration"})),
    "typescript": _Spec(
        _ts("tree_sitter_typescript", "language_typescript"),
        _TS_DEFS,
        frozenset({"class_declaration", "abstract_class_declaration", "internal_module"}),
    ),
    "tsx": _Spec(
        _ts("tree_sitter_typescript", "language_tsx"),
        _TS_DEFS,
        frozenset({"class_declaration", "abstract_class_declaration", "internal_module"}),
    ),
    "rust": _Spec(
        _ts("tree_sitter_rust"),
        {
            "use_declaration": "use",
            "function_item": "fn",
            "struct_item": "struct",
            "enum_item": "enum",
            "trait_item": "trait",
            "impl_item": "impl",
            "mod_item": "mod",
            "type_item": "type",
            "const_item": "const",
            "static_item": "static",
        },
        frozenset({"impl_item", "trait_item", "mod_item"}),
    ),
    "go": _Spec(
        _ts("tree_sitter_go"),
        {
            "import_declaration": "import",
            "function_declaration": "func",
            "method_declaration": "method",
            "type_declaration": "type",
        },
    ),
    "c": _Spec(_ts("tree_sitter_c"), _C_DEFS),
    "cpp": _Spec(
        _ts("tree_sitter_cpp"),
        _C_DEFS
        | {
            "namespace_definition": "namespace",
            "class_specifier": "class",
            "struct_specifier": "struct",
        },
        frozenset({"namespace_definition"}),
    ),
    "java": _Spec(
        _ts("tree_sitter_java"),
        {
            "import_declaration": "import",
            "class_declaration": "class",
            "interface_declaration": "interface",
            "enum_declaration": "enum",
            "record_declaration": "record",
            "method_declaration": "method",
            "constructor_declaration": "constructor",
        },
        frozenset({"class_declaration", "interface_declaration", "enum_declaration"}),
    ),
    "ruby": _Spec(
        _ts("tree_sitter_ruby"),
        {"method": "def", "singleton_method": "def", "class": "class", "module": "module"},
        frozenset({"class", "module"}),
    ),
    "bash": _Spec(_ts("tree_sitter_bash"), {"function_definition": "function"}),
    "zig": _Spec(
        _ts("tree_sitter_zig"),
        {"function_declaration": "fn", "test_declaration": "test"},
    ),
    # Elixir defs are ``call`` nodes; _walk classifies them by keyword.
    "elixir": _Spec(_ts("tree_sitter_elixir"), {}),
}

_EXT_TO_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "tsx",
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "tsx",
    ".rs": "rust",
    ".go": "go",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".java": "java",
    ".rb": "ruby",
    ".sh": "bash",
    ".bash": "bash",
    ".zig": "zig",
    ".ex": "elixir",
    ".exs": "elixir",
}

_ELIXIR_DEFS = {
    "defmodule": "defmodule",
    "defprotocol": "defprotocol",
    "defimpl": "defimpl",
    "def": "def",
    "defp": "defp",
    "defmacro": "defmacro",
    "defmacrop": "defmacrop",
    "defstruct": "defstruct",
    "defguard": "defguard",
    "defguardp": "defguardp",
    "alias": "alias",
    "import": "import",
    "require": "require",
    "use": "use",
}
_ELIXIR_CONTAINERS = {"defmodule", "defprotocol", "defimpl"}

_parsers: dict[str, Any] = {}


def _parser(language: str) -> Any:
    """Cached ``tree_sitter.Parser`` per language (fail-open on build issues)."""
    if language not in _parsers:
        from tree_sitter import Language, Parser

        _parsers[language] = Parser(Language(_SPECS[language].loader()))
    return _parsers[language]


def _signature(node: tree_sitter.Node, source: bytes) -> str:
    """The node's first line, or its text up to the body block when longer."""
    body = node.child_by_field_name("body")
    if body is not None and body.start_byte > node.start_byte:
        raw = source[node.start_byte : body.start_byte].decode("utf-8", errors="replace")
    else:
        raw = source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
        raw = raw.split("\n", 1)[0]
    sig = " ".join(raw.split()).rstrip(":{;").strip()
    if sig.endswith(" do"):  # elixir blocks: "defmodule Foo do" → "defmodule Foo"
        sig = sig[:-3]
    if len(sig) > MAX_SIG_CHARS:
        sig = sig[: MAX_SIG_CHARS - 1] + "…"
    return sig


def _walk(
    node: tree_sitter.Node,
    language: str,
    source: bytes,
    depth: int,
    out: list[tuple[int, str, int, int]],
) -> None:
    spec = _SPECS.get(language)
    for child in node.children:
        node_type = child.type
        kind: str | None = None
        container = False
        if language == "elixir" and node_type == "call":
            keyword = _elixir_keyword(child, source)
            if keyword in _ELIXIR_DEFS:
                kind = _ELIXIR_DEFS[keyword]
                container = keyword in _ELIXIR_CONTAINERS
        elif language in ("javascript", "typescript", "tsx") and node_type == "lexical_declaration":
            if _declares_function(child):
                kind = "function"
        elif spec is not None:
            kind = spec.defs.get(node_type)
            container = node_type in spec.containers
        if kind is not None:
            out.append(
                (depth, _signature(child, source), child.start_point[0] + 1, child.end_point[0] + 1)
            )
            if container:
                _walk(child, language, source, depth + 1, out)
            continue
        _walk(child, language, source, depth, out)


def _elixir_keyword(call_node: tree_sitter.Node, source: bytes) -> str:
    """The keyword of an elixir ``call`` node (``def``, ``defmodule``, …)."""
    first = call_node.children[0] if call_node.children else None
    if first is None or first.type != "identifier":
        return ""
    return source[first.start_byte : first.end_byte].decode("utf-8", errors="replace")


def _declares_function(node: tree_sitter.Node) -> bool:
    """``const f = () => …`` style declarations: keep only function values."""
    for child in node.children:
        if child.type != "variable_declarator":
            continue
        value = child.child_by_field_name("value")
        if value is not None and value.type in (
            "arrow_function",
            "function",
            "function_expression",
        ):
            return True
    return False


def _resolve(ctx: ToolContext, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ctx.cwd / p


class IndexTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="index",
            description=(
                "Show the skeleton of a source file: imports, types, and "
                "function/method signatures with line ranges. Use it before "
                "read on large files, then read only the ranges you need."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Source file path"},
                },
                "required": ["path"],
            },
        )

    async def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        path = _resolve(ctx, str(args.get("path", "")))
        if not path.exists():
            return ToolResult(f"error: no such file: {args.get('path')}", is_error=True)
        if path.is_dir():
            return ToolResult(f"error: is a directory: {args.get('path')}", is_error=True)

        language = _EXT_TO_LANGUAGE.get(path.suffix.lower())
        if language is None:
            return ToolResult(f"unsupported file type for index: {path.suffix or '(none)'}")

        try:
            source = path.read_bytes()
            tree = _parser(language).parse(source)
        except Exception as e:
            return ToolResult(f"index unavailable for {path.name}: {e}", is_error=True)

        items: list[tuple[int, str, int, int]] = []
        _walk(tree.root_node, language, source, 0, items)

        total_lines = source.count(b"\n")
        if source and not source.endswith(b"\n"):
            total_lines += 1
        header = f"{path.name} — {language}, {total_lines} lines"
        if not items:
            return ToolResult(f"{header}\n(no definitions found)")

        lines = [header]
        for depth, sig, start, end in items[:MAX_ITEMS]:
            span = f"L{start}" if start == end else f"L{start}-{end}"
            lines.append(f"{'  ' * depth}{sig} ({span})")
        if len(items) > MAX_ITEMS:
            lines.append(f"… {len(items) - MAX_ITEMS} more items (use read with offset/limit)")
        return ToolResult("\n".join(lines))


def make_tool() -> Tool:
    return IndexTool()
