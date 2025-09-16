#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified progress line renderer with default dividers.

Behavioral notes:
- Prints a divider line BEFORE the first row (at init) and AFTER close().
- Keeps columns aligned across tools, even if FPS is hidden.
- Label column is fixed width (10 chars).

This is used by face_extract.py, openface3.py, mediapipe.py, etc.
"""
import sys, time, shutil

PARTIALS = "▏▎▍▌▋▊▉"
BAR_WIDTH = 26  # exact length requested
DIV = "=" * 103  # matches the requested visual width


def _render_bar(ratio: float, width: int) -> str:
    r = max(0.0, min(1.0, float(ratio)))
    full = int(width * r)
    frac = width * r - full
    bar = "█" * min(full, width)
    if full < width:
        idx = int(frac * len(PARTIALS))
        if idx > 0:
            bar += PARTIALS[min(idx, len(PARTIALS) - 1)]
            full += 1
        if full < width:
            bar += " " * (width - full)
    return bar


def _fmt_eta(secs: float) -> str:
    secs = max(0, int(secs))
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class ProgressBar:
    def __init__(
        self,
        label: str,
        total: int,
        show_fail_label: bool = True,
        stream = sys.stdout,
        min_interval: float = 0.05,
        fail_label: str = "FAIL",
        show_fps: bool = True,
        show_dividers: bool = True,
    ):
        self.label = f"{str(label)[:10]:<10}"
        self.total = max(1, int(total))
        self.stream = stream
        self.show_fail_label = bool(show_fail_label)
        self.fail_label = str(fail_label)
        self.show_fps = bool(show_fps)
        self._t0 = time.time()
        self._last_draw = 0.0
        self._min_interval = float(min_interval)
        self._prev_len = 0
        self._final = False
        self._printed_div_start = False
        self._printed_div_end = False
        if show_dividers:
            self.stream.write(DIV + "\n")
            self._printed_div_start = True
            self.stream.flush()

    def _line(self, done: int, fails: int | None = None, fps: float | None = None) -> str:
        now = time.time()
        ratio = min(1.0, max(0.0, float(done) / float(self.total)))
        bar = _render_bar(ratio, BAR_WIDTH)
        pct = 100.0 * ratio
        eta = _fmt_eta(int((now - self._t0) * (1 - ratio) / max(1e-9, ratio))) if done > 0 else "00:00"
        if fps is None:
            fps_val = done / max(1e-6, now - self._t0)
        else:
            fps_val = float(fps)
        # Keep fixed-width columns (11 for FAIL label+count; 8 for FPS field)
        fail_str = f"{self.fail_label} {int(fails):6d}" if (self.show_fail_label and fails is not None) else (" " * 11)
        fps_str = (f"{fps_val:4.0f} FPS" if self.show_fps else (" " * 8))
        return (
            f"{self.label} |{bar}| {pct:6.1f}% {done:8d}/{self.total:<8d} "
            f"{fail_str} {fps_str} ETA {eta}"
        )

    def update(self, done: int, fails: int | None = None, fps: float | None = None) -> None:
        if self._final:
            return
        now = time.time()
        if (now - self._last_draw) < self._min_interval and done < self.total:
            return
        self._last_draw = now
        line = self._line(done, fails=fails, fps=fps)
        pad = max(0, self._prev_len - len(line))
        self.stream.write("\r" + line + (" " * pad))
        self._prev_len = len(line)
        if done >= self.total:
            self.stream.write("\n")
            self._final = True
            self.stream.flush()

    def close(self) -> None:
        if not self._final:
            self.update(self.total)
        if not self._printed_div_end:
            self.stream.write(DIV + "\n")
            self._printed_div_end = True
            self.stream.flush()
