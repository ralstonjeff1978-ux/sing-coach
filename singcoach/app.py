"""Application entry point.

    python -m singcoach
"""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from . import config
from .ui import theme


def main(argv: list[str] | None = None) -> int:
    config.ensure_dirs()

    QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName(config.APP_NAME)
    app.setStyleSheet(theme.STYLESHEET)

    from .ui.main_window import MainWindow

    window = MainWindow()
    window.show()

    # Reopen whatever was last practised, so the app lands where you left it.
    if config.load_settings().last_song_hash:
        try:
            from .library.song import Song

            song = Song.load(config.load_settings().last_song_hash)
            if song.is_ready_to_practice:
                window._activate(song)
        except (FileNotFoundError, ValueError, OSError):
            pass

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
