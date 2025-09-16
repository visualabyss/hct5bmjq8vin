#!/usr/bin/env python3
import sys, time, shutil

PARTIALS = "▏▎▍▌▋▊▉"
BAR_WIDTH = 26  # exact length requested

def _render_bar(ratio: float, width: int) -> str:
    r = max(0.0, min(1.0, float(ratio)))
    full = int(width * r)
    frac = width * r - full
    bar = "█" * min(full, width)
    if full < width:
        idx = int(frac * len(PARTIALS))
        if idx > 0:
            bar += PARTIALS[min(idx, len(PARTIALS)-1)]
            full += 1
    if full < width:
        bar += " " * (width - full)
    return bar

def _fmt_eta(secs):
    secs = max(0, int(secs))
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

class ProgressBar:
    def __init__(self, label, total, show_fail_label=True, stream=sys.stdout, min_interval=0.05,
                 fail_label="FAIL", show_fps=True):
        self.label = f"{label:<10}"[:10]
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

    def _line(self, done, fails=None, fps=None):
        now = time.time()
        ratio = min(1.0, max(0.0, float(done)/float(self.total)))
        bar = _render_bar(ratio, BAR_WIDTH)
        pct = 100.0 * ratio
        eta = _fmt_eta(int((now - self._t0) * (1 - ratio) / max(1e-9, ratio))) if done>0 else "00:00"

        if fps is None:
            fps_val = done / max(1e-6, now - self._t0)
        else:
            fps_val = float(fps)

        fail_str = f"{self.fail_label} {int(fails):6d}" if (self.show_fail_label and fails is not None) else " " * 11
        fps_str  = (f"{fps_val:4.0f} FPS" if self.show_fps else "         ")

        return f"{self.label} |{bar}| {pct:6.1f}% {done:8d}/{self.total:<8d}  {fail_str}  {fps_str}    ETA {eta}"

    def update(self, done, fails=None, fps=None):
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

    def close(self):
        if not self._final:
            self.update(self.total)
