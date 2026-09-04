"""Multimodal input: image/audio/PDF attachments.

Files are sniffed by magic bytes first, then extension; loaded attachments
are base64-encoded (20 MB cap) and sent as OpenAI-compatible content parts —
images as ``image_url`` parts, audio/PDF as ``file`` parts (the OpenRouter
convention; both are defined in :mod:`lecode.providers.types`). The model
catalog's per-model modality flags gate submission: a clear error names the
unsupported modality; unknown models fail open.

Attachments collect in an :class:`AttachmentStore` on the app (``/add``, or
``@path`` tokens in a submission that resolve to media files) and attach to
the next user message only, then clear.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from lecode.providers.types import ContentPart, FilePart, ImagePart

if TYPE_CHECKING:
    from lecode.providers.catalog import Catalog

#: Hard cap per attachment.
MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024

MEDIA_KINDS = ("image", "audio", "pdf")

#: User-message content: plain text, or text + attachment parts.
MessageContent = str | list[ContentPart]

#: Extension → media kind.
_EXTENSION_KINDS = {
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".gif": "image",
    ".webp": "image",
    ".mp3": "audio",
    ".wav": "audio",
    ".m4a": "audio",
    ".ogg": "audio",
    ".flac": "audio",
    ".pdf": "pdf",
}

#: Media kind (+ extension for jpg) → MIME type.
_MIMES = {
    ("image", ".png"): "image/png",
    ("image", ".jpg"): "image/jpeg",
    ("image", ".jpeg"): "image/jpeg",
    ("image", ".gif"): "image/gif",
    ("image", ".webp"): "image/webp",
    ("audio", ".mp3"): "audio/mpeg",
    ("audio", ".wav"): "audio/wav",
    ("audio", ".m4a"): "audio/mp4",
    ("audio", ".ogg"): "audio/ogg",
    ("audio", ".flac"): "audio/flac",
    ("pdf", ".pdf"): "application/pdf",
}

#: Magic bytes → (media kind, extension to assume for the MIME lookup).
_MAGIC: list[tuple[bytes, str, str]] = [
    (b"\x89PNG\r\n\x1a\n", "image", ".png"),
    (b"\xff\xd8\xff", "image", ".jpg"),
    (b"GIF8", "image", ".gif"),
    (b"%PDF", "pdf", ".pdf"),
    (b"ID3", "audio", ".mp3"),
    (b"\xff\xfb", "audio", ".mp3"),
    (b"OggS", "audio", ".ogg"),
    (b"fLaC", "audio", ".flac"),
]

#: Bytes read for magic-byte sniffing.
_SNIFF_BYTES = 16


@dataclass(frozen=True)
class Attachment:
    """One loaded attachment: sniffed kind, MIME, size, base64 payload."""

    path: Path
    media_kind: str  # image | audio | pdf
    mime: str
    size_bytes: int
    data: str  # base64

    @property
    def data_uri(self) -> str:
        return f"data:{self.mime};base64,{self.data}"


def _sniff_magic(header: bytes) -> tuple[str, str] | None:
    for magic, kind, ext in _MAGIC:
        if header.startswith(magic):
            return kind, ext
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return "image", ".webp"
    if header.startswith(b"RIFF") and header[8:12] == b"WAVE":
        return "audio", ".wav"
    if header[4:8] == b"ftyp" and len(header) >= 8:
        return "audio", ".m4a"
    return None


def sniff(path: Path | str) -> str | None:
    """The media kind for ``path`` (magic bytes first, then extension)."""
    path = Path(path)
    try:
        with path.open("rb") as f:
            header = f.read(_SNIFF_BYTES)
    except OSError:
        header = b""
    magic = _sniff_magic(header)
    if magic is not None:
        return magic[0]
    return _EXTENSION_KINDS.get(path.suffix.lower())


def load_attachment(path: Path | str) -> Attachment:
    """Load and base64-encode a media file; clear errors otherwise."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {path}")
    kind = sniff(path)
    if kind is None:
        raise ValueError(f"unsupported attachment type (want image/audio/PDF): {path.name}")
    size = path.stat().st_size
    if size > MAX_ATTACHMENT_BYTES:
        raise ValueError(
            f"attachment too large: {path.name} ({format_size(size)} > "
            f"{format_size(MAX_ATTACHMENT_BYTES)})"
        )
    ext = path.suffix.lower()
    mime = _MIMES.get((kind, ext)) or _MIMES[(kind, _default_ext(kind))]
    data = base64.b64encode(path.read_bytes()).decode()
    return Attachment(path=path, media_kind=kind, mime=mime, size_bytes=size, data=data)


def _default_ext(kind: str) -> str:
    return {"image": ".png", "audio": ".mp3", "pdf": ".pdf"}[kind]


def format_size(size: int) -> str:
    """``2048`` → ``2 KB``."""
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    return f"{max(1, size // 1024)} KB"


def to_content_parts(attachments: list[Attachment]) -> list[ContentPart]:
    """OpenAI-compatible parts: images as ``image_url``, audio/PDF as ``file``."""
    parts: list[ContentPart] = []
    for attachment in attachments:
        if attachment.media_kind == "image":
            parts.append(
                ImagePart(
                    type="image_url",
                    image_url={"url": attachment.data_uri},
                )
            )
        else:
            parts.append(
                FilePart(
                    type="file",
                    file={
                        "filename": attachment.path.name,
                        "file_data": attachment.data_uri,
                    },
                )
            )
    return parts


def check_modalities(attachments: list[Attachment], model: str, catalog: Catalog) -> str | None:
    """``None`` when ``model`` accepts every attachment; else an error naming
    the unsupported modalities. Unknown models fail open (note in the docstring
    of the caller: the provider will reject what it cannot take)."""
    from lecode.providers.catalog import AmbiguousModelError, ModelNotFoundError

    try:
        info = catalog.get(model)
    except (ModelNotFoundError, AmbiguousModelError):
        return None  # fail open: unknown model, let the provider decide
    supported = set(info.modalities.input)
    missing = sorted({a.media_kind for a in attachments} - supported)
    if not missing:
        return None
    names = ", ".join(a.path.name for a in attachments if a.media_kind in missing)
    return (
        f"model {info.id} does not support {'/'.join(missing)} input "
        f"(supports: {', '.join(sorted(supported))}) — drop {names} or switch model"
    )


def describe_content(content: MessageContent) -> str:
    """Display text for a user message: text plus an attachment summary."""
    if isinstance(content, str):
        return content
    text = " ".join(p.get("text", "") for p in content if p.get("type") == "text")
    count = sum(1 for p in content if p.get("type") in ("image_url", "file"))
    return f"{text}  📎 {count} attachment(s)" if count else text


class AttachmentStore:
    """Pending attachments for the next user submission."""

    def __init__(self) -> None:
        self._items: list[Attachment] = []

    def __len__(self) -> int:
        return len(self._items)

    def list(self) -> list[Attachment]:
        return list(self._items)

    def add(self, attachment: Attachment) -> None:
        self._items.append(attachment)

    def drop(self, ref: str) -> Attachment | None:
        """Drop by 1-based index or file name; ``None`` when not found."""
        if ref.isdigit():
            index = int(ref) - 1
            if 0 <= index < len(self._items):
                return self._items.pop(index)
            return None
        for index, attachment in enumerate(self._items):
            if attachment.path.name == ref or str(attachment.path) == ref:
                return self._items.pop(index)
        return None

    def clear(self) -> int:
        """Drop everything; returns how many were pending."""
        count = len(self._items)
        self._items.clear()
        return count


def extract_attachment_refs(text: str, cwd: Path, store: AttachmentStore) -> str:
    """Pull ``@path`` tokens that resolve to media files into ``store``.

    This is how the ``@`` picker attaches files: the completer inserts the
    path text, and submission parsing converts it. Tokens that don't resolve
    to a sniffable media file are left untouched (``@agent`` mentions among
    them); failed loads (e.g. over the cap) also stay as plain text.
    """
    kept: list[str] = []
    for word in text.split():
        if word.startswith("@") and len(word) > 1:
            candidate = Path(word[1:])
            if not candidate.is_absolute():
                candidate = cwd / candidate
            if candidate.is_file() and sniff(candidate) is not None:
                try:
                    store.add(load_attachment(candidate))
                    continue
                except (OSError, ValueError):
                    pass  # leave the token as plain text
        kept.append(word)
    return " ".join(kept)
