#!/usr/bin/env python3
import sys, time, shutil

PARTIALS = "▏▎▍▌▋▊▉"
TARGET_BAR = 40           # 20% shorter target
MIN_BAR    = 10
DIV_FALLBACK = 103

def _render_bar(ratio: float, width: int) -> str:
    r = max(0.0, min(1.0, float(ratio)))
    full = int(width * r)
    frac = width * r - full
    bar = "█" * min(full, width)
    if full < width:
        idx = int(frac * len(PARTIALS))
        if idx > 0:
            bar += PARTIALS[idx - 1]
            full += 1
    if full < width:
        bar += " " * (width - full)
    return bar

class ProgressBar:
    """
    Single-line, no-wrap progress:
      - fits to terminal columns
      - clears leftover chars from previous draw
      - redraw rate-limited (~25 Hz)
    """
    def __init__(self, label, total, show_fail_label=True, stream=sys.stdout, min_interval=0.04):
        self.label = f"{label:<10}"[:10]
        self.total = max(1, int(total))
        self.stream = stream
        self.show_fail_label = show_fail_label
        self._t0 = time.time()
        self._last_done = -1
        self._last_t = 0.0
        self._min_interval = float(min_interval)
        self._closed = False
        self._prev_len = 0

        cols = shutil.get_terminal_size(fallback=(100, 24)).columns
        # compute base length with empty bar to size the bar
        fail_str = " " * 11
        base = f"{self.label} || {0:6.1f}% {0:8d}/{self.total:<8d}  {fail_str}  {0:4.0f} FPS    ETA 00:00"
        base_len = len(base)
        bar_space = max(MIN_BAR, min(TARGET_BAR, cols - base_len - 1))
        self.bar_width = bar_space

        div_len = max(10, min(cols, DIV_FALLBACK))
        self._div = "=" * div_len

        self.stream.write(self._div + "\n")
        self.stream.flush()

    def _line(self, done, fails, fps_override=None) -> str:
        ratio = done / self.total
        bar   = _render_bar(ratio, self.bar_width)
        pct   = f"{ratio*100:6.1f}%"
        count = f"{done:8d}/{self.total:<8d}"
        elapsed = max(1e-6, time.time() - self._t0)
        fps = fps_override if fps_override is not None else (done / elapsed if done > 0 else 0.0)
        eta_s = int((self.total - done) / fps) if fps > 0 else 0
        eta   = f"{eta_s//3600:02d}:{(eta_s%3600)//60:02d}"
        fail_str = f"FAIL {fails:6d}" if (self.show_fail_label and fails is not None) else " " * 11
        return f"{self.label} |{bar}| {pct} {count}  {fail_str}  {fps:4.0f} FPS    ETA {eta}"

    def update(self, done, fails=None, fps=None):
        if self._closed: return
        done = max(0, min(int(done), self.total))
        now = time.time()
        if done == self._last_done: return
        if done < self.total and (now - self._last_t) < self._min_interval: return
        self._last_done = done
        self._last_t = now

        line = self._line(done, fails, fps)
        # hard trim in case of weird terminals
        cols = shutil.get_terminal_size(fallback=(100,24)).columns
        if len(line) >= cols:
            line = line[:max(0, cols-1)]

        # carriage return, write line, then pad spaces to erase leftovers
        pad = " " * max(0, self._prev_len - len(line))
        self.stream.write("\r" + line + pad)
        self._prev_len = len(line)
        if done >= self.total:
            self.stream.write("\n" + self._div + "\n")
        self.stream.flush()

    def close(self):
        if self._closed: return
        if self._last_done < self.total:
            self.update(self.total, None)
        self._closed = True
