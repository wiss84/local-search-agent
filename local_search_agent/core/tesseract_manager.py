"""
Tesseract OCR detector for the Local Search Agent framework.

Tesseract is an optional dependency that significantly speeds up OCR on
scanned/image-based PDFs (~1 sec/page vs minutes with RapidOCR).

If Tesseract is not installed, the pipeline falls back to RapidOCR + ONNXRuntime
automatically — no crash, no error, just slower OCR on scanned documents.

Installation instructions (one-time, manual):
  Windows : https://github.com/UB-Mannheim/tesseract/wiki
            Download and run the installer. Make sure to check
            "Add to PATH" during installation.
  Linux   : sudo apt install tesseract-ocr        (Ubuntu/Debian)
            sudo dnf install tesseract             (Fedora/RHEL)
            sudo pacman -S tesseract               (Arch)
  macOS   : brew install tesseract

After installation, restart the application — it will be detected automatically.
"""

from __future__ import annotations

import logging
import shutil
from typing import Optional

logger = logging.getLogger(__name__)


def get_tesseract_cmd() -> Optional[str]:
    """
    Return the path to the tesseract executable if available on PATH, else None.

    Assign this to pytesseract.pytesseract.tesseract_cmd before any calls.
    """
    return shutil.which("tesseract")


def is_available() -> bool:
    """Return True if tesseract is installed and on PATH."""
    return get_tesseract_cmd() is not None


class TesseractManager:
    """
    Detect and configure Tesseract OCR availability.

    No downloading, no installation — just detects what the user has installed.
    If Tesseract is not found, the OCR pipeline falls back to RapidOCR silently.
    """

    def __init__(self):
        self._cmd: Optional[str] = None

    def ensure(self) -> Optional[str]:
        """
        Return the tesseract executable path if available, else None.

        Logs a one-time informational message if not found so the user
        knows they can install it to speed up scanned PDF ingestion.
        """
        if self._cmd is not None:
            return self._cmd

        cmd = get_tesseract_cmd()
        if cmd:
            self._cmd = cmd
            logger.info("Tesseract detected at: %s", cmd)
        else:
            logger.info(
                "Tesseract not found on PATH — scanned PDFs will use RapidOCR (slower). "
                "To enable faster OCR, install Tesseract and restart:\n"
                "  Windows : https://github.com/UB-Mannheim/tesseract/wiki\n"
                "  Linux   : sudo apt install tesseract-ocr\n"
                "  macOS   : brew install tesseract"
            )
        return cmd

    def configure_pytesseract(self) -> bool:
        """
        Point pytesseract at the detected binary.

        Returns True if configured successfully, False otherwise.
        """
        cmd = self.ensure()
        if cmd is None:
            return False
        try:
            import pytesseract  # noqa: PLC0415

            pytesseract.pytesseract.tesseract_cmd = cmd
            return True
        except ImportError:
            logger.warning("pytesseract is not installed. Run: pip install pytesseract")
            return False


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_MANAGER: Optional[TesseractManager] = None


def get_tesseract_manager() -> TesseractManager:
    """Return the module-level TesseractManager singleton."""
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = TesseractManager()
    return _MANAGER
