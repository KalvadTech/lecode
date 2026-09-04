"""Tests for multimodal attachments: sniffing, parts, modalities, wiring."""

from __future__ import annotations

import base64

import pytest
from tests.test_tui_app import make_app, make_blocking_app, wait_for

from lecode.agent.tools.base import ToolContext, ToolRegistry
from lecode.agent.tools.read import ReadTool
from lecode.config.models import Config
from lecode.multimodal import (
    MAX_ATTACHMENT_BYTES,
    AttachmentStore,
    check_modalities,
    describe_content,
    extract_attachment_refs,
    format_size,
    load_attachment,
    sniff,
    to_content_parts,
)
from lecode.permission import PermissionChecker
from lecode.providers.catalog import Catalog

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
JPG = b"\xff\xd8\xff\xe0" + b"\x00" * 16
PDF = b"%PDF-1.4\n" + b"\x00" * 16
MP3 = b"ID3\x04\x00" + b"\x00" * 16

TEXT_ONLY_MODEL = "deepseek/deepseek-r1"
IMAGE_MODEL = "openai/gpt-5-mini"
PDF_MODEL = "anthropic/claude-sonnet-4"
AUDIO_MODEL = "google/gemini-2.5-pro"


@pytest.fixture
def catalog() -> Catalog:
    return Catalog.default()


def _write(tmp_path, name: str, data: bytes):
    path = tmp_path / name
    path.write_bytes(data)
    return path


# -- sniff --------------------------------------------------------------------------


def test_sniff_magic_bytes(tmp_path):
    assert sniff(_write(tmp_path, "a.png", PNG)) == "image"
    assert sniff(_write(tmp_path, "b.jpg", JPG)) == "image"
    assert sniff(_write(tmp_path, "c.pdf", PDF)) == "pdf"
    assert sniff(_write(tmp_path, "d.mp3", MP3)) == "audio"


def test_sniff_magic_beats_extension(tmp_path):
    # a .png file whose bytes are a PDF sniffs as pdf
    assert sniff(_write(tmp_path, "disguised.png", PDF)) == "pdf"


def test_sniff_riff_containers(tmp_path):
    webp = b"RIFF" + b"\x10\x00\x00\x00" + b"WEBP" + b"\x00" * 8
    wav = b"RIFF" + b"\x10\x00\x00\x00" + b"WAVE" + b"\x00" * 8
    assert sniff(_write(tmp_path, "a.webp", webp)) == "image"
    assert sniff(_write(tmp_path, "a.wav", wav)) == "audio"


def test_sniff_extension_fallback(tmp_path):
    assert sniff(_write(tmp_path, "plain.pdf", b"not really a pdf")) == "pdf"
    assert sniff(_write(tmp_path, "plain.flac", b"not flac")) == "audio"


def test_sniff_unsupported(tmp_path):
    assert sniff(_write(tmp_path, "notes.txt", b"hello")) is None
    assert sniff(_write(tmp_path, "archive.zip", b"PK\x03\x04")) is None


# -- load_attachment --------------------------------------------------------------------


def test_load_attachment_base64_roundtrip(tmp_path):
    attachment = load_attachment(_write(tmp_path, "a.png", PNG))
    assert attachment.media_kind == "image"
    assert attachment.mime == "image/png"
    assert attachment.size_bytes == len(PNG)
    assert base64.b64decode(attachment.data) == PNG
    assert attachment.data_uri.startswith("data:image/png;base64,")


def test_load_attachment_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="no such file"):
        load_attachment(tmp_path / "nope.png")


def test_load_attachment_unsupported_type(tmp_path):
    with pytest.raises(ValueError, match="unsupported attachment type"):
        load_attachment(_write(tmp_path, "notes.txt", b"hello"))


def test_load_attachment_size_cap(tmp_path):
    big = _write(tmp_path, "big.png", PNG + b"\x00" * MAX_ATTACHMENT_BYTES)
    with pytest.raises(ValueError, match="attachment too large"):
        load_attachment(big)


def test_format_size():
    assert format_size(2048) == "2 KB"
    assert format_size(5 * 1024 * 1024) == "5.0 MB"


# -- content parts ---------------------------------------------------------------------


def test_content_parts_image(tmp_path):
    parts = to_content_parts([load_attachment(_write(tmp_path, "a.png", PNG))])
    assert parts == [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{base64.b64encode(PNG).decode()}"},
        }
    ]


def test_content_parts_audio_and_pdf(tmp_path):
    attachments = [
        load_attachment(_write(tmp_path, "a.mp3", MP3)),
        load_attachment(_write(tmp_path, "b.pdf", PDF)),
    ]
    parts = to_content_parts(attachments)
    assert parts[0] == {
        "type": "file",
        "file": {
            "filename": "a.mp3",
            "file_data": f"data:audio/mpeg;base64,{base64.b64encode(MP3).decode()}",
        },
    }
    assert parts[1]["type"] == "file"
    assert parts[1]["file"]["filename"] == "b.pdf"
    assert parts[1]["file"]["file_data"].startswith("data:application/pdf;base64,")


# -- modality checks ------------------------------------------------------------------


def test_modalities_supported(tmp_path, catalog):
    image = load_attachment(_write(tmp_path, "a.png", PNG))
    assert check_modalities([image], IMAGE_MODEL, catalog) is None
    pdf = load_attachment(_write(tmp_path, "b.pdf", PDF))
    assert check_modalities([pdf], PDF_MODEL, catalog) is None
    audio = load_attachment(_write(tmp_path, "c.mp3", MP3))
    assert check_modalities([audio], AUDIO_MODEL, catalog) is None


def test_modalities_unsupported_names_kind_and_file(tmp_path, catalog):
    image = load_attachment(_write(tmp_path, "a.png", PNG))
    error = check_modalities([image], TEXT_ONLY_MODEL, catalog)
    assert "does not support image input" in error
    assert "a.png" in error
    pdf = load_attachment(_write(tmp_path, "b.pdf", PDF))
    assert "does not support pdf input" in check_modalities([pdf], IMAGE_MODEL, catalog)


def test_modalities_unknown_model_fails_open(tmp_path, catalog):
    image = load_attachment(_write(tmp_path, "a.png", PNG))
    assert check_modalities([image], "local/finetune-v9", catalog) is None


# -- AttachmentStore --------------------------------------------------------------------


def _attachment(tmp_path, name: str):
    return load_attachment(_write(tmp_path, name, PNG))


def test_store_add_list_drop(tmp_path):
    store = AttachmentStore()
    first, second = _attachment(tmp_path, "a.png"), _attachment(tmp_path, "b.png")
    store.add(first)
    store.add(second)
    assert len(store) == 2
    assert store.drop("2") == second  # by 1-based index
    assert store.drop("a.png") == first  # by file name
    assert len(store) == 0


def test_store_drop_unknown_ref(tmp_path):
    store = AttachmentStore()
    store.add(_attachment(tmp_path, "a.png"))
    assert store.drop("5") is None
    assert store.drop("zzz.png") is None
    assert len(store) == 1


def test_store_clear_counts(tmp_path):
    store = AttachmentStore()
    assert store.clear() == 0
    store.add(_attachment(tmp_path, "a.png"))
    store.add(_attachment(tmp_path, "b.png"))
    assert store.clear() == 2
    assert len(store) == 0


# -- @path extraction -----------------------------------------------------------------


def test_extract_attachment_refs_converts_media(tmp_path):
    _write(tmp_path, "img.png", PNG)
    store = AttachmentStore()
    text = extract_attachment_refs("look at @img.png please", tmp_path, store)
    assert text == "look at please"
    assert len(store) == 1
    assert store.list()[0].path.name == "img.png"


def test_extract_attachment_refs_leaves_agents_and_missing(tmp_path):
    store = AttachmentStore()
    text = extract_attachment_refs("@plan check @missing.png", tmp_path, store)
    assert text == "@plan check @missing.png"
    assert len(store) == 0


# -- /add /drop /drop-all handlers --------------------------------------------------------


async def test_add_handler_attaches_and_reports(tmp_path, monkeypatch):
    _write(tmp_path, "img.png", PNG)
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/add img.png")
    assert len(app.attachments) == 1
    assert "📎 img.png (image," in out.getvalue()


async def test_add_handler_errors_inline(tmp_path, monkeypatch):
    _write(tmp_path, "notes.txt", b"hello")
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/add missing.png notes.txt")
    rendered = out.getvalue()
    assert "no such file" in rendered
    assert "unsupported attachment type" in rendered
    assert len(app.attachments) == 0
    await app.handle_command("/add")
    assert "usage: /add" in out.getvalue()


async def test_drop_handler(tmp_path, monkeypatch):
    _write(tmp_path, "a.png", PNG)
    _write(tmp_path, "b.png", PNG)
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/add a.png b.png")
    out.truncate(0)
    out.seek(0)
    await app.handle_command("/drop")  # lists pending
    assert "1. a.png" in out.getvalue() and "2. b.png" in out.getvalue()
    await app.handle_command("/drop 1")
    assert "dropped a.png" in out.getvalue()
    await app.handle_command("/drop 9")
    assert "no such attachment" in out.getvalue()
    assert len(app.attachments) == 1


async def test_drop_all_handler(tmp_path, monkeypatch):
    _write(tmp_path, "a.png", PNG)
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/drop-all")
    assert "(no pending attachments)" in out.getvalue()
    await app.handle_command("/add a.png")
    await app.handle_command("/drop-all")
    assert "dropped 1 attachment(s)" in out.getvalue()
    assert len(app.attachments) == 0


# -- submit flow ----------------------------------------------------------------------


async def test_submit_sends_content_parts(tmp_path, monkeypatch):
    _write(tmp_path, "img.png", PNG)
    config = Config()
    config.llm.model = IMAGE_MODEL
    app, provider, out = make_app(tmp_path, monkeypatch, [{"text": "it's a png"}], config=config)
    await app.handle_command("/add img.png")
    await app._submit("what is this")
    await app._turn_task
    message = provider.requests[0]["messages"][-1]
    assert message["role"] == "user"
    content = message["content"]
    assert content[0] == {"type": "text", "text": "what is this"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert len(app.attachments) == 0  # cleared after send
    assert "> what is this  📎 img.png" in out.getvalue()


async def test_submit_modality_error_blocks_send(tmp_path, monkeypatch):
    _write(tmp_path, "img.png", PNG)
    config = Config()
    config.llm.model = TEXT_ONLY_MODEL
    app, provider, out = make_app(tmp_path, monkeypatch, [], config=config)
    await app.handle_command("/add img.png")
    await app._submit("what is this")
    rendered = out.getvalue()
    assert f"model {TEXT_ONLY_MODEL} does not support image input" in rendered
    assert provider.requests == []  # not sent
    assert len(app.attachments) == 1  # kept for /drop or a model switch


async def test_submit_at_path_attachment(tmp_path, monkeypatch):
    _write(tmp_path, "shot.png", PNG)
    config = Config()
    config.llm.model = IMAGE_MODEL
    app, provider, out = make_app(tmp_path, monkeypatch, [{"text": "described"}], config=config)
    await app._submit("describe @shot.png")
    await app._turn_task
    message = provider.requests[0]["messages"][-1]
    assert message["content"][0] == {"type": "text", "text": "describe"}
    assert message["content"][1]["type"] == "image_url"
    assert "📎 shot.png (image," in out.getvalue()


async def test_queued_message_with_attachments(tmp_path, monkeypatch):
    _write(tmp_path, "img.png", PNG)
    config = Config()
    config.llm.model = IMAGE_MODEL
    app, provider, out = make_blocking_app(tmp_path, monkeypatch, config=config)
    await app._submit("first")
    await wait_for(lambda: len(provider.requests) == 1)
    await app._submit("@img.png queued pic")
    provider.blocked = False
    provider.release.set()
    await wait_for(lambda: not app._turn_running())
    last = provider.requests[-1]["messages"][-1]
    assert last["content"][0] == {"type": "text", "text": "queued pic"}
    assert last["content"][1]["type"] == "image_url"
    assert "📎 1 attachment(s)" in out.getvalue()  # follow-up echo


# -- read tool -------------------------------------------------------------------------


def _read_ctx(tmp_path, model: str) -> ToolContext:
    config = Config()
    config.llm.model = model
    checker = PermissionChecker(config, mode="yolo", cwd=tmp_path)
    return ToolContext(cwd=tmp_path, config=config, permission_checker=checker, auto_approve=True)


async def test_read_image_part_when_model_supports(tmp_path):
    _write(tmp_path, "img.png", PNG)
    registry = ToolRegistry([ReadTool()])
    message, result = await registry.dispatch_result(
        "c1", "read", '{"path": "img.png"}', _read_ctx(tmp_path, IMAGE_MODEL)
    )
    assert not result.is_error
    content = message["content"]
    assert content[0]["type"] == "text"
    assert "image file: img.png" in content[0]["text"]
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


async def test_read_image_metadata_note_when_unsupported(tmp_path):
    _write(tmp_path, "img.png", PNG)
    registry = ToolRegistry([ReadTool()])
    message, result = await registry.dispatch_result(
        "c1", "read", '{"path": "img.png"}', _read_ctx(tmp_path, TEXT_ONLY_MODEL)
    )
    assert isinstance(message["content"], str)
    assert "not enabled for this model" in message["content"]
    assert result.metadata["image"] is True


# -- describe_content ------------------------------------------------------------------


def test_describe_content():
    assert describe_content("plain") == "plain"
    parts = [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": "data:..."}},
    ]
    assert describe_content(parts) == "look  📎 1 attachment(s)"
