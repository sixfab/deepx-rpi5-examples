"""
tui_input.py — Raw keyboard input for arrow key navigation.
Handles: Up/Down arrows, Enter, 'q' to quit.
Works on Linux/macOS (Raspberry Pi compatible).
"""
import sys
import termios
import tty


def get_keypress() -> str:
    """
    Read a single keypress from stdin without requiring Enter.
    Returns: 'up' | 'down' | 'enter' | 'q' | 'other'
    """
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == '\x1b':
            ch2 = sys.stdin.read(2)
            if ch2 == '[A':
                return 'up'
            if ch2 == '[B':
                return 'down'
            return 'other'
        elif ch in ('\r', '\n'):
            return 'enter'
        elif ch in ('q', 'Q'):
            return 'q'
        elif ch == '\x03':
            raise KeyboardInterrupt
        else:
            return 'other'
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def wait_any_key() -> None:
    """Block until the user hits any key."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
