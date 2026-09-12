"""Visual language for the app.

Design constraints came from what this thing is actually for. You look at it
while singing — from a few feet away, in motion, with your attention mostly on
the music. So:

* **Dark by default.** A bright screen in a dim room is fatiguing, and the
  pitch trace reads better against dark.
* **Accuracy is colour, and colour alone carries no critical meaning.** Being
  flat is also shown by position and by words, because roughly one man in
  twelve has some red/green colour deficiency and "your note is red" would be
  invisible to them.
* **One loud element at a time.** The verdict is the only thing allowed to
  shout. Everything else stays quiet so the eye goes where it should.
"""

from __future__ import annotations

from PySide6.QtGui import QColor, QFont

# -- surfaces ---------------------------------------------------------------

BG = QColor(14, 16, 21)
PANEL = QColor(22, 25, 32)
PANEL_LIGHT = QColor(31, 35, 44)
BORDER = QColor(46, 52, 64)

TEXT = QColor(226, 232, 240)
TEXT_DIM = QColor(138, 148, 166)
TEXT_FAINT = QColor(86, 95, 112)

# -- accuracy ---------------------------------------------------------------
# Chosen to stay distinguishable under the common forms of colour blindness:
# the in-tune colour is a cyan-leaning green and the error colours differ in
# lightness as well as hue.

ON_PITCH = QColor(52, 211, 153)
NEAR = QColor(250, 204, 21)
OFF = QColor(248, 113, 113)
WRONG_OCTAVE = QColor(167, 139, 250)

# -- highway ----------------------------------------------------------------

TARGET_NOTE = QColor(59, 130, 246)
TARGET_NOTE_ACTIVE = QColor(96, 165, 250)
GHOST_CONTOUR = QColor(148, 163, 184, 90)     # the original singer's shape
HARMONY_LINE = QColor(196, 181, 253, 120)
NO_TARGET = QColor(71, 85, 105, 60)
#: Shading over the stretches where the original vocal is sounding — "sing
#: here". Deliberately faint: it is a backdrop, not a thing to look at.
VOCAL_BAND = QColor(59, 130, 246, 26)
PLAYHEAD = QColor(248, 250, 252, 200)
GRIDLINE = QColor(38, 43, 54)
GRIDLINE_OCTAVE = QColor(58, 66, 82)

ACCENT = QColor(56, 189, 248)
RECORD = QColor(239, 68, 68)


def accuracy_color(cents: float | None, tolerance: float = 20.0) -> QColor:
    """Map a pitch error onto the accuracy palette."""
    if cents is None:
        return TEXT_FAINT
    magnitude = abs(cents)
    if magnitude > 600:
        return WRONG_OCTAVE
    if magnitude <= tolerance:
        return ON_PITCH
    if magnitude <= tolerance * 2.5:
        return NEAR
    return OFF


def mono_font(size: int = 11, bold: bool = False) -> QFont:
    f = QFont("Cascadia Mono", size)
    f.setStyleHint(QFont.StyleHint.Monospace)
    f.setBold(bold)
    return f


def ui_font(size: int = 10, bold: bool = False) -> QFont:
    f = QFont("Segoe UI", size)
    f.setBold(bold)
    return f


STYLESHEET = f"""
QWidget {{
    background-color: {BG.name()};
    color: {TEXT.name()};
    font-family: 'Segoe UI';
    font-size: 13px;
}}
QFrame#panel {{
    background-color: {PANEL.name()};
    border: 1px solid {BORDER.name()};
    border-radius: 8px;
}}
QLabel#heading {{
    color: {TEXT_DIM.name()};
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
}}
QPushButton {{
    background-color: {PANEL_LIGHT.name()};
    border: 1px solid {BORDER.name()};
    border-radius: 6px;
    padding: 7px 14px;
    color: {TEXT.name()};
}}
QPushButton:hover  {{ background-color: {BORDER.name()}; }}
QPushButton:pressed {{ background-color: {PANEL.name()}; }}
QPushButton:disabled {{ color: {TEXT_FAINT.name()}; }}
QPushButton#primary {{
    background-color: {ACCENT.name()};
    border: none;
    color: #06121c;
    font-weight: 600;
}}
QPushButton#primary:hover {{ background-color: #7dd3fc; }}
QPushButton#record {{
    background-color: {RECORD.name()};
    border: none;
    color: white;
    font-weight: 600;
}}
QPushButton:checked {{
    background-color: {ACCENT.name()};
    color: #06121c;
    font-weight: 600;
}}
QSlider::groove:horizontal {{
    height: 4px;
    background: {BORDER.name()};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {TEXT.name()};
    width: 14px;
    height: 14px;
    margin: -6px 0;
    border-radius: 7px;
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT.name()};
    border-radius: 2px;
}}
QComboBox {{
    background-color: {PANEL_LIGHT.name()};
    border: 1px solid {BORDER.name()};
    border-radius: 6px;
    padding: 5px 10px;
}}
QComboBox QAbstractItemView {{
    background-color: {PANEL_LIGHT.name()};
    selection-background-color: {ACCENT.name()};
    selection-color: #06121c;
}}
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{
    width: 17px; height: 17px;
    border: 1px solid {BORDER.name()};
    border-radius: 4px;
    background: {PANEL_LIGHT.name()};
}}
QCheckBox::indicator:checked {{
    background: {ACCENT.name()};
    border-color: {ACCENT.name()};
}}
QProgressBar {{
    border: 1px solid {BORDER.name()};
    border-radius: 6px;
    background: {PANEL.name()};
    text-align: center;
    height: 18px;
}}
QProgressBar::chunk {{ background-color: {ACCENT.name()}; border-radius: 5px; }}
QToolTip {{
    background-color: {PANEL_LIGHT.name()};
    color: {TEXT.name()};
    border: 1px solid {BORDER.name()};
    padding: 5px;
}}
"""
