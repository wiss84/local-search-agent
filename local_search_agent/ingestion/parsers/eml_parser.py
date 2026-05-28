"""
EML parser for the Local Search Agent ingestion pipeline.

Uses Python's stdlib email module — no third-party dependency.

Strategy:
- Extracts standard headers: From, To, Cc, Subject, Date
- Extracts plain text body (text/plain parts preferred over text/html)
- Falls back to HTML body stripped of tags if no plain text part exists
- Attachments are noted by filename only (content not extracted)
- Nested multipart messages are walked recursively
- Output is a clean Markdown document with headers as a metadata block
"""

from __future__ import annotations

import email
import email.policy
import logging
import os
import re
from email.header import decode_header, make_header
from typing import Optional

from local_search_agent.core.document_node import DocumentNode
from local_search_agent.ingestion.cleaner import clean
from local_search_agent.ingestion.parser import BaseParser, ParserError

logger = logging.getLogger(__name__)


def _decode_header_value(raw: Optional[str]) -> str:
    """Decode an encoded email header value to a plain string."""
    if not raw:
        return ""
    try:
        return str(make_header(decode_header(raw)))
    except Exception:
        return raw


def _strip_html_tags(html: str) -> str:
    """Minimal HTML tag stripper for email body fallback."""
    text = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    text = re.sub(r"<p\s*/?>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_body(msg: email.message.Message) -> tuple[str, list[str]]:
    """
    Walk a Message and extract:
    - plain text body (preferred) or HTML fallback
    - list of attachment filenames

    Returns (body_text, attachment_names).
    """
    plain_parts: list[str] = []
    html_parts: list[str] = []
    attachments: list[str] = []

    for part in msg.walk():
        content_type = part.get_content_type()
        disposition = part.get_content_disposition() or ""

        # Attachments — record filename only
        if "attachment" in disposition:
            filename = part.get_filename()
            if filename:
                attachments.append(_decode_header_value(filename))
            continue

        if content_type == "text/plain":
            charset = part.get_content_charset() or "utf-8"
            try:
                payload = part.get_payload(decode=True)
                if payload:
                    plain_parts.append(payload.decode(charset, errors="replace"))
            except Exception as e:
                logger.debug("Failed to decode text/plain part: %s", e)

        elif content_type == "text/html":
            charset = part.get_content_charset() or "utf-8"
            try:
                payload = part.get_payload(decode=True)
                if payload:
                    html_parts.append(payload.decode(charset, errors="replace"))
            except Exception as e:
                logger.debug("Failed to decode text/html part: %s", e)

    if plain_parts:
        body = "\n\n".join(plain_parts)
    elif html_parts:
        body = _strip_html_tags("\n".join(html_parts))
    else:
        body = ""

    return body, attachments


class EMLParser(BaseParser):
    """Parse .eml email files into a structured Markdown DocumentNode."""

    @property
    def supported_extensions(self) -> frozenset[str]:
        return frozenset({".eml"})

    def parse(
        self,
        source_path: str,
        workspace: str,
        title: Optional[str] = None,
    ) -> DocumentNode:
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"EML file not found: {source_path!r}")

        logger.info("Parsing EML: %s", source_path)

        try:
            with open(source_path, "rb") as f:
                msg = email.message_from_binary_file(f, policy=email.policy.compat32)
        except Exception as e:
            raise ParserError(source_path, f"EML read failed: {e}", original=e)

        # Decode headers
        subject = _decode_header_value(msg.get("Subject"))
        from_ = _decode_header_value(msg.get("From"))
        to_ = _decode_header_value(msg.get("To"))
        cc_ = _decode_header_value(msg.get("Cc"))
        date_ = _decode_header_value(msg.get("Date"))

        # Use subject as document title if not overridden
        if title is None:
            title = subject or os.path.splitext(os.path.basename(source_path))[0]

        # Extract body and attachments
        try:
            body, attachments = _extract_body(msg)
        except Exception as e:
            raise ParserError(source_path, f"EML body extraction failed: {e}", original=e)

        # Build Markdown output
        sections: list[str] = []

        # Metadata block
        meta_lines = ["## Email Metadata\n"]
        if date_:
            meta_lines.append(f"**Date**: {date_}")
        if from_:
            meta_lines.append(f"**From**: {from_}")
        if to_:
            meta_lines.append(f"**To**: {to_}")
        if cc_:
            meta_lines.append(f"**Cc**: {cc_}")
        if subject:
            meta_lines.append(f"**Subject**: {subject}")
        sections.append("\n".join(meta_lines))

        # Body
        if body.strip():
            sections.append(f"## Body\n\n{body.strip()}")

        # Attachments
        if attachments:
            att_list = "\n".join(f"- {name}" for name in attachments)
            sections.append(f"## Attachments\n\n{att_list}")

        raw_text = "\n\n".join(sections)

        if not raw_text.strip():
            raise ParserError(source_path, "EML produced empty output.")

        cleaned_text = clean(raw_text)

        return DocumentNode.from_file(
            source_path=source_path,
            text=cleaned_text,
            workspace=workspace,
            title=title,
        )
