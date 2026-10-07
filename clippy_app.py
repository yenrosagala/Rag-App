"""
clippy_app.py — Laptop RAG App (Clippy edition)

A local-first RAG assistant that lives on your desktop as a draggable
Clippy-style widget instead of a browser tab. Click him to open a small
ask-popup near him (like app.py); he answers in his own fading speech
bubble - that's the main way you talk to him. Right-click for Settings,
Chat History (a read-only scrollback window you open on demand), an
Avatar Style/Shape picker, Reset Position, or Quit.

Beyond plain questions, a few simple commands are recognized in the ask
popup (typed with or without a leading "/"):
    summarize [topic]   - summarize the docs, optionally about a topic
    sources             - list files currently in the data folder
    clear               - clear chat history
    help                - show the command list

Pipeline underneath is unchanged from a "real" RAG app:
    modules/ingest.py  -> load + chunk documents
    modules/rag.py     -> sentence-transformers embeddings, Chroma store,
                           retrieval + LLM answer (Ollama or OpenAI)
    modules/engine.py  -> settings persistence + a blocking facade the
                           UI drives from background threads

Run with:
    python clippy_app.py
"""
from __future__ import annotations

import math
import queue
import random
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from modules.engine import RagEngine, Settings

TRANSPARENT_KEY = "#ff00ff"  # matches app.py's transparent key
CLIPPY_SIZE = 300
CLIPPY_HEIGHT = 260
AVATAR_SCALE = 1.35
ASSET_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)) / "assets"

MARU_FILES = {
    "idle": "maru_idle.png",    # neutral / slightly unimpressed resting face
    "happy": "maru_happy.png",  # eyes closed, content
    "talk": "maru_talk.png",    # mouth open, mid-reaction
}

ANIME_MOODS = ("idle", "happy", "talk")
USER_IMAGE_EXTS = ("png", "gif", "ppm", "pgm")


def _resolve_user_images(prefix: str) -> dict[str, str]:
    """Resolve per-mood frames <prefix>_<mood>.<ext>, falling back to one
    single <prefix>.<ext> image shared by every mood. tk.PhotoImage reads
    PNG, GIF, PPM and PGM natively, so any of those formats work."""
    frames = {}
    for mood in ANIME_MOODS:
        for ext in USER_IMAGE_EXTS:
            cand = ASSET_DIR / f"{prefix}_{mood}.{ext}"
            if cand.is_file():
                frames[mood] = cand.name
                break
    single = next(
        (f"{prefix}.{ext}" for ext in USER_IMAGE_EXTS if (ASSET_DIR / f"{prefix}.{ext}").is_file()),
        None,
    )
    fill = single or next(iter(frames.values()), None)
    if fill:
        for mood in ANIME_MOODS:
            frames.setdefault(mood, fill)
    return frames


def _load_photo(path: Path) -> tk.PhotoImage:
    """Plain-Tkinter PNG loader (no Pillow dependency at runtime - the
    assets are already pre-shrunk to widget size)."""
    return tk.PhotoImage(file=str(path))


# --------------------------------------------------------------------------
# Clippy drawing (pure Canvas — no image assets needed / nothing for
# PyInstaller to lose track of). Ported from app.py's ClippyAvatar so both
# apps present the same hand-drawn character: same bent-paperclip silhouette,
# same darker-grey palette, and the same blink/bounce idle animation instead
# of a static pose. The "maru" shape is the one exception - it draws a PNG
# image (see MARU_FILES) instead of vector shapes, since Maru is hand-drawn
# artwork rather than something that translates well to canvas primitives.
# --------------------------------------------------------------------------
class ClippyAvatar:
    """Animated paperclip character with blinking eyes, a gentle bounce,
    and an on-canvas speech bubble that fades after a few seconds - visually
    identical to the one in app.py. Supports a few selectable color styles."""

    STYLES = {
        "classic":  {"body": "#7a7a7a", "brow": "#555"},
        "silver":   {"body": "#c9cfd6", "brow": "#5a6068"},
        "navy":     {"body": "#3a5a7a", "brow": "#24384d"},
        "rose":     {"body": "#b06a7a", "brow": "#7a3f4a"},
        "gold":     {"body": "#c9a227", "brow": "#8a6d15"},
        "forest":   {"body": "#3f7a4e", "brow": "#254a2f"},
        "charcoal": {"body": "#2e2e2e", "brow": "#000000"},
        "sky":      {"body": "#5aa9d6", "brow": "#2f6b91"},
        "plum":     {"body": "#7a4a8a", "brow": "#4d2c58"},
        "copper":   {"body": "#b5651d", "brow": "#7a4213"},
    }

    # Selectable body silhouettes. Each maps to a "_draw_<shape>" method below.
    SHAPES = ["paperclip", "circle", "star", "diamond", "cat", "robot", "heart", "ghost", "maru", "anime"]

    def __init__(self, canvas: tk.Canvas, body_y: int = 115, scale: float = AVATAR_SCALE,
                 style: str = "classic", shape: str = "paperclip", on_bubble_resize=None):
        self.canvas = canvas
        self.blink_state = False
        self.bounce_offset = 0
        self.bubble_text = ""
        self.bubble_timer_id = None
        self.body_y = body_y
        self.scale = scale
        self._tick_count = 0
        # Optional callback(bubble_height_px) -> body_y, provided by the
        # parent widget so it can resize/reposition the window to fit a
        # long speech bubble instead of clipping it. See ClippyWidget.
        self.on_bubble_resize = on_bubble_resize
        self._pending_bubble = None  # (text, wrap_width, bubble_h) for the next _draw_bubble call
        self.set_style(style, redraw=False)
        self.set_shape(shape, redraw=False)
        self._draw()
        self._animate()

    def set_style(self, style: str, redraw: bool = True):
        colors = self.STYLES.get(style, self.STYLES["classic"])
        self.style = style if style in self.STYLES else "classic"
        self.BODY_COLOR = colors["body"]
        self.BROW_COLOR = colors["brow"]
        if redraw:
            self._draw()

    def set_shape(self, shape: str, redraw: bool = True):
        self.shape = shape if shape in self.SHAPES else "paperclip"
        if redraw:
            self._draw()

    def _s(self, x, y):
        """Scale a raw (x, y) point outward from the clip's anchor so the
        whole character grows without redoing every coordinate by hand."""
        ox, oy = 82, self.body_y
        return ox + (x - ox) * self.scale, oy + (y - self.body_y) * self.scale

    def _draw(self):
        c = self.canvas
        c.delete("clippy")
        self._apply_bubble_sizing()
        y = self.body_y + self.bounce_offset
        draw_fn = getattr(self, f"_draw_{self.shape}", self._draw_paperclip)
        draw_fn(c, y)
        if self.bubble_text:
            self._draw_bubble()

    # -- shared face parts (eyes / eyebrows), reused by every body shape ---
    def _draw_face(self, c, eye_y, cx=82, offset=7, half_w=3.5, brow_drop=8):
        lx0, ly0 = self._s(cx - offset - half_w, eye_y - 4)
        lx1, ly1 = self._s(cx - offset + half_w, eye_y + 4)
        rx0, ry0 = self._s(cx + offset - half_w, eye_y - 4)
        rx1, ry1 = self._s(cx + offset + half_w, eye_y + 4)
        if self.blink_state:
            c.create_line(lx0, (ly0 + ly1) / 2, lx1, (ly0 + ly1) / 2,
                           width=2, fill="black", tags="clippy")
            c.create_line(rx0, (ry0 + ry1) / 2, rx1, (ry0 + ry1) / 2,
                           width=2, fill="black", tags="clippy")
        else:
            c.create_oval(lx0, ly0, lx1, ly1, fill="white", outline="black", tags="clippy")
            plx0, ply0 = self._s(cx - offset - 1.5, eye_y - 2)
            plx1, ply1 = self._s(cx - offset + 1.5, eye_y + 2)
            c.create_oval(plx0, ply0, plx1, ply1, fill="black", tags="clippy")
            c.create_oval(rx0, ry0, rx1, ry1, fill="white", outline="black", tags="clippy")
            prx0, pry0 = self._s(cx + offset - 1.5, eye_y - 2)
            prx1, pry1 = self._s(cx + offset + 1.5, eye_y + 2)
            c.create_oval(prx0, pry0, prx1, pry1, fill="black", tags="clippy")

        # eyebrows for personality
        blx0, bly0 = self._s(cx - offset - 5, eye_y - brow_drop)
        blx1, bly1 = self._s(cx - offset + 4, eye_y - brow_drop - 2)
        brx0, bry0 = self._s(cx + offset - 4, eye_y - brow_drop - 2)
        brx1, bry1 = self._s(cx + offset + 5, eye_y - brow_drop)
        c.create_line(blx0, bly0, blx1, bly1, width=2, fill=self.BROW_COLOR, tags="clippy")
        c.create_line(brx0, bry0, brx1, bry1, width=2, fill=self.BROW_COLOR, tags="clippy")

    @staticmethod
    def _star_points(cx, cy, outer, inner, points=5):
        pts = []
        for i in range(points * 2):
            r = outer if i % 2 == 0 else inner
            ang = math.pi / 2 + i * math.pi / points
            pts.append((cx + r * math.cos(ang), cy - r * math.sin(ang)))
        return pts

    # -- body shapes ---------------------------------------------------
    def _draw_paperclip(self, c, y):
        # bent paperclip body (two nested rounded loops)
        raw_pts = [(70, y - 55), (70, y + 25), (95, y + 40), (95, y - 40),
                   (75, y - 50), (75, y + 15), (90, y + 25), (90, y - 35)]
        pts = [c for p in raw_pts for c in self._s(*p)]
        c.create_line(*pts, width=max(6, round(6 * self.scale)), fill=self.BODY_COLOR,
                       capstyle="round", joinstyle="round", smooth=True, tags="clippy")
        self._draw_face(c, eye_y=y - 45)

    def _draw_circle(self, c, y):
        x0, y0 = self._s(47, y - 80)
        x1, y1 = self._s(117, y - 10)
        c.create_oval(x0, y0, x1, y1, fill=self.BODY_COLOR, outline=self.BROW_COLOR,
                       width=2, tags="clippy")
        self._draw_face(c, eye_y=y - 48, offset=9, half_w=4)

    def _draw_star(self, c, y):
        pts = self._star_points(82, y - 45, outer=42, inner=18)
        flat = [v for p in pts for v in self._s(*p)]
        c.create_polygon(*flat, fill=self.BODY_COLOR, outline=self.BROW_COLOR,
                          width=2, joinstyle="round", tags="clippy")
        self._draw_face(c, eye_y=y - 45, offset=8, half_w=3)

    def _draw_diamond(self, c, y):
        raw_pts = [(82, y - 85), (122, y - 45), (82, y - 5), (42, y - 45)]
        flat = [v for p in raw_pts for v in self._s(*p)]
        c.create_polygon(*flat, fill=self.BODY_COLOR, outline=self.BROW_COLOR,
                          width=2, joinstyle="round", tags="clippy")
        self._draw_face(c, eye_y=y - 50, offset=7, half_w=3.5)

    def _draw_cat(self, c, y):
        # ears (behind the head)
        lx0, ly0 = self._s(56, y - 65)
        lx1, ly1 = self._s(68, y - 92)
        lx2, ly2 = self._s(80, y - 65)
        c.create_polygon(lx0, ly0, lx1, ly1, lx2, ly2, fill=self.BODY_COLOR,
                          outline=self.BROW_COLOR, width=2, tags="clippy")
        rx0, ry0 = self._s(84, y - 65)
        rx1, ry1 = self._s(96, y - 92)
        rx2, ry2 = self._s(108, y - 65)
        c.create_polygon(rx0, ry0, rx1, ry1, rx2, ry2, fill=self.BODY_COLOR,
                          outline=self.BROW_COLOR, width=2, tags="clippy")
        # round head
        hx0, hy0 = self._s(52, y - 70)
        hx1, hy1 = self._s(112, y - 10)
        c.create_oval(hx0, hy0, hx1, hy1, fill=self.BODY_COLOR, outline=self.BROW_COLOR,
                       width=2, tags="clippy")
        # nose
        nx0, ny0 = self._s(78, y - 32)
        nx1, ny1 = self._s(86, y - 32)
        nx2, ny2 = self._s(82, y - 26)
        c.create_polygon(nx0, ny0, nx1, ny1, nx2, ny2, fill=self.BROW_COLOR, tags="clippy")
        self._draw_face(c, eye_y=y - 45, offset=9, half_w=3.5)

    _shape_photo_cache: dict[str, tk.PhotoImage] = {}  # keyed by filename, shared across instances

    def _shape_photo(self, files: dict, mood: str) -> tk.PhotoImage:
        cache = ClippyAvatar._shape_photo_cache
        filename = files.get(mood, files["idle"])
        if filename not in cache:
            cache[filename] = _load_photo(ASSET_DIR / filename)
        return cache[filename]

    def _image_mood(self) -> str:
        """Talking while there's something to say, otherwise a slow
        idle/happy blink so the character doesn't sit frozen."""
        if self.bubble_text:
            return "talk"
        return "happy" if self.blink_state else "idle"

    def _draw_image_shape(self, files: dict, c, y):
        try:
            photo = self._shape_photo(files, self._image_mood())
        except tk.TclError:
            # Asset missing - fall back to the paperclip rather than a blank widget.
            self._draw_paperclip(c, y)
            return
        # Unlike the vector shapes, these PNGs are already pre-sized in
        # real pixels to fit the widget - so anchor using plain canvas
        # coordinates, NOT self._s() (which would re-apply AVATAR_SCALE on
        # top of that and push the image's top off the canvas).
        cx = CLIPPY_SIZE / 2
        # body_y sits a fixed distance above the canvas's bottom edge (see
        # ClippyPet._resize_for_bubble, which keeps that gap constant even
        # as the window grows taller to fit a long bubble) - so anchoring
        # this many px below y always lands right at the canvas bottom,
        # regardless of window resizing.
        bottom = y + (CLIPPY_HEIGHT - 170) - 4
        c.create_image(cx, bottom, image=photo, anchor="s", tags="clippy")

    def _draw_maru(self, c, y):
        self._draw_image_shape(MARU_FILES, c, y)

    _anime_files_resolved: dict[str, str] | None = None

    def _draw_anime(self, c, y):
        if ClippyAvatar._anime_files_resolved is None:
            ClippyAvatar._anime_files_resolved = _resolve_user_images("anime")
        if ClippyAvatar._anime_files_resolved:
            self._draw_image_shape(ClippyAvatar._anime_files_resolved, c, y)
        else:
            self._draw_paperclip(c, y)

    def _draw_robot(self, c, y):
        x0, y0 = self._s(50, y - 75)
        x1, y1 = self._s(114, y - 15)
        c.create_rectangle(x0, y0, x1, y1, fill=self.BODY_COLOR, outline=self.BROW_COLOR,
                            width=2, tags="clippy")
        # antenna
        ax0, ay0 = self._s(82, y - 75)
        ax1, ay1 = self._s(82, y - 95)
        c.create_line(ax0, ay0, ax1, ay1, width=2, fill=self.BROW_COLOR, tags="clippy")
        bx0, by0 = self._s(78, y - 99)
        bx1, by1 = self._s(86, y - 91)
        c.create_oval(bx0, by0, bx1, by1, fill=self.BROW_COLOR, tags="clippy")
        # speaker-grille mouth
        mx0, my0 = self._s(68, y - 28)
        mx1, my1 = self._s(96, y - 22)
        c.create_rectangle(mx0, my0, mx1, my1, fill="white", outline=self.BROW_COLOR, tags="clippy")
        self._draw_face(c, eye_y=y - 52, offset=9, half_w=3.5)

    def _draw_heart(self, c, y):
        lx0, ly0 = self._s(56, y - 72)
        lx1, ly1 = self._s(84, y - 42)
        rx0, ry0 = self._s(80, y - 72)
        rx1, ry1 = self._s(108, y - 42)
        tri = [(58, y - 54), (106, y - 54), (82, y - 6)]
        flat = [v for p in tri for v in self._s(*p)]
        c.create_polygon(*flat, fill=self.BODY_COLOR, outline=self.BROW_COLOR,
                          width=2, joinstyle="round", tags="clippy")
        c.create_oval(lx0, ly0, lx1, ly1, fill=self.BODY_COLOR, outline=self.BROW_COLOR,
                       width=2, tags="clippy")
        c.create_oval(rx0, ry0, rx1, ry1, fill=self.BODY_COLOR, outline=self.BROW_COLOR,
                       width=2, tags="clippy")
        self._draw_face(c, eye_y=y - 55, offset=8, half_w=3)

    def _draw_ghost(self, c, y):
        cx, top_r = 82, 32
        top_cy = y - 58
        pts = []
        # rounded top (semicircle)
        for i in range(9):
            ang = math.pi - i * (math.pi / 8)
            pts.append((cx + top_r * math.cos(ang), top_cy - top_r * math.sin(ang)))
        # wavy bottom, right to left
        pts += [(cx + top_r, top_cy + 28), (cx + top_r - 8, top_cy + 38),
                (cx + top_r - 16, top_cy + 26), (cx + top_r - 24, top_cy + 38),
                (cx + top_r - 32, top_cy + 26), (cx + top_r - 40, top_cy + 38),
                (cx + top_r - 48, top_cy + 26), (cx - top_r, top_cy + 28)]
        flat = [v for p in pts for v in self._s(*p)]
        c.create_polygon(*flat, fill=self.BODY_COLOR, outline=self.BROW_COLOR,
                          width=2, tags="clippy")
        self._draw_face(c, eye_y=top_cy - 2, offset=8, half_w=3.5, brow_drop=6)

    # Truncate only as a last resort for very long answers - the full text
    # is always still available in Chat History, so this just keeps the
    # on-screen bubble from growing unboundedly.
    _MAX_BUBBLE_CHARS = 700

    def _apply_bubble_sizing(self):
        """Measure the (possibly multi-line) bubble text and, if a resize
        callback was given, ask the parent window to grow/shrink to fit -
        so the bubble is sized to its content instead of clipping it. Runs
        before the avatar body is drawn so self.body_y is correct either way."""
        if not self.bubble_text:
            self._pending_bubble = None
            if self.on_bubble_resize:
                self.body_y = self.on_bubble_resize(0)
            return

        text = self.bubble_text
        if len(text) > self._MAX_BUBBLE_CHARS:
            text = text[: self._MAX_BUBBLE_CHARS - 30].rstrip() + "…  (full answer in Chat History)"

        wrap_width = int(230 * self.scale)
        padding_x, padding_y = 12, 10

        c = self.canvas
        probe_id = c.create_text(-9999, -9999, text=text, width=wrap_width,
                                  font=("Segoe UI", 8), anchor="nw")
        bbox = c.bbox(probe_id)
        c.delete(probe_id)
        text_w = (bbox[2] - bbox[0]) if bbox else wrap_width
        text_h = (bbox[3] - bbox[1]) if bbox else 16

        bubble_w = max(text_w + padding_x * 2, int(120 * self.scale))
        bubble_h = text_h + padding_y * 2
        self._pending_bubble = (text, wrap_width, padding_x, padding_y, bubble_w, bubble_h)

        if self.on_bubble_resize:
            self.body_y = self.on_bubble_resize(bubble_h)

    def _draw_bubble(self):
        if not self._pending_bubble:
            return
        text, wrap_width, padding_x, padding_y, bubble_w, bubble_h = self._pending_bubble
        c = self.canvas

        # Matches the original bubble's vertical offset from the avatar's
        # anchor point, so the tail still points at roughly the same spot.
        anchor_gap = int(115 * self.scale) - 60
        left = 10
        right = left + bubble_w
        rect_bottom = self.body_y - anchor_gap
        rect_top = rect_bottom - bubble_h

        c.create_rectangle(left, rect_top, right, rect_bottom, fill="white",
                            outline="#999", width=1, tags="clippy")
        tail_cx = left + min(45, bubble_w * 0.25)
        c.create_polygon(tail_cx, rect_bottom, tail_cx + 15, rect_bottom,
                          tail_cx + 5, rect_bottom + 13,
                          fill="white", outline="#999", tags="clippy")
        c.create_text(left + padding_x, rect_top + padding_y, text=text,
                       width=wrap_width, font=("Segoe UI", 8), anchor="nw", tags="clippy")

    def _animate(self):
        if self._blink_tick():
            self.blink_state = not self.blink_state
        # Smooth sine bob instead of a rigid sawtooth; taller + faster while
        # talking so the pet visibly perks up mid-sentence. _s() re-applies
        # AVATAR_SCALE on top of this offset for the actual pixel motion.
        amp = 8 if self.bubble_text else 4
        self.bounce_offset = round(amp * math.sin(self._tick_count * 0.22))
        self._draw()
        self.canvas.after(180, self._animate)

    def _blink_tick(self):
        # blink roughly every ~2.5s, stay closed for one tick
        self._tick_count += 1
        return self._tick_count % 14 == 0

    def say(self, text, duration_ms: int | None = 5000):
        """Show text in the speech bubble. duration_ms=None means it stays
        up indefinitely - no auto-clear timer - until something else calls
        say() again (e.g. the next question) or _clear_bubble() directly."""
        self.bubble_text = text
        self._draw()
        if self.bubble_timer_id:
            self.canvas.after_cancel(self.bubble_timer_id)
            self.bubble_timer_id = None
        if duration_ms is not None:
            self.bubble_timer_id = self.canvas.after(duration_ms, self._clear_bubble)

    def _clear_bubble(self):
        self.bubble_text = ""
        self.bubble_timer_id = None
        self._draw()


class AskPopup(tk.Toplevel):
    """A small, single-line entry that pops up next to Clippy and closes
    itself on submit or focus-out — same ephemeral pattern as app.py's
    _open_ask_popup, instead of a persistent chat window."""

    def __init__(self, master: "ClippyPet"):
        super().__init__(master.root)
        self.pet = master
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self.configure(bg="#333333")

        wx, wy = master.root.winfo_x(), master.root.winfo_y()
        popup_w, popup_h = 260, 40
        self.geometry(f"{popup_w}x{popup_h}+{wx}+{max(wy - popup_h - 6, 0)}")

        self.entry = tk.Entry(self, font=("Segoe UI", 10))
        self.entry.pack(fill="both", expand=True, padx=4, pady=4)
        self.entry.focus_set()
        self.entry.bind("<Return>", self._submit)
        self.entry.bind("<Escape>", self._cancel)
        self.bind("<FocusOut>", self._cancel)

    def _submit(self, event=None):
        query = self.entry.get().strip()
        self.pet.ask_popup = None
        self.destroy()
        if query:
            self.pet.send_question(query)

    def _cancel(self, event=None):
        self.pet.ask_popup = None
        self.destroy()


class SettingsWindow(tk.Toplevel):
    def __init__(self, master: "ClippyPet"):
        super().__init__(master.root)
        self.pet = master
        self.title("Laptop RAG — Settings")
        self.resizable(False, False)
        self.attributes("-topmost", True)
        s = master.engine.settings
        pad = dict(padx=10, pady=4, sticky="w")

        row = 0
        tk.Label(self, text="Documents folder").grid(row=row, column=0, **pad)
        self.data_dir_var = tk.StringVar(value=s.data_dir)
        tk.Entry(self, textvariable=self.data_dir_var, width=42).grid(
            row=row, column=1, padx=(0, 4), pady=4)
        tk.Button(self, text="Browse...", command=self._browse).grid(row=row, column=2, padx=(0, 10))

        row += 1
        tk.Label(self, text="LLM backend").grid(row=row, column=0, **pad)
        self.backend_var = tk.StringVar(value=s.backend)
        ttk.Combobox(self, textvariable=self.backend_var, values=["ollama", "openai"],
                     state="readonly", width=15).grid(row=row, column=1, sticky="w", pady=4)

        row += 1
        tk.Label(self, text="Model name").grid(row=row, column=0, **pad)
        self.model_var = tk.StringVar(value=s.model)
        tk.Entry(self, textvariable=self.model_var, width=42).grid(row=row, column=1, pady=4)

        row += 1
        tk.Label(self, text="OpenAI API key (if backend=openai)").grid(row=row, column=0, **pad)
        self.key_var = tk.StringVar(value=s.openai_api_key)
        tk.Entry(self, textvariable=self.key_var, width=42, show="*").grid(row=row, column=1, pady=4)

        row += 1
        tk.Label(self, text="Chunks to retrieve (k)").grid(row=row, column=0, **pad)
        self.k_var = tk.IntVar(value=s.k)
        tk.Spinbox(self, from_=1, to=10, textvariable=self.k_var, width=5).grid(
            row=row, column=1, sticky="w", pady=4)

        row += 1
        self.status_var = tk.StringVar(value="")
        tk.Label(self, textvariable=self.status_var, fg="#555", wraplength=380,
                 justify="left").grid(row=row, column=0, columnspan=3, padx=10, pady=(6, 0), sticky="w")

        row += 1
        btns = tk.Frame(self)
        btns.grid(row=row, column=0, columnspan=3, pady=10)
        self.rebuild_btn = tk.Button(btns, text="🔄 Rebuild index", command=self._rebuild)
        self.rebuild_btn.pack(side="left", padx=6)
        tk.Button(btns, text="Save", command=self._save).pack(side="left", padx=6)
        tk.Button(btns, text="Close", command=self.destroy).pack(side="left", padx=6)

    def _browse(self):
        d = filedialog.askdirectory(initialdir=self.data_dir_var.get() or ".")
        if d:
            self.data_dir_var.set(d)

    def _apply_to_settings(self):
        s = self.pet.engine.settings
        s.data_dir = self.data_dir_var.get().strip() or s.data_dir
        s.backend = self.backend_var.get()
        s.model = self.model_var.get().strip() or s.model
        s.openai_api_key = self.key_var.get().strip()
        s.k = int(self.k_var.get())
        s.save()

    def _save(self):
        self._apply_to_settings()
        self.status_var.set("Settings saved.")

    def _rebuild(self):
        self._apply_to_settings()
        self.rebuild_btn.configure(state="disabled")
        self.status_var.set("Working...")

        def ui(fn):
            # The settings window can be destroyed while the worker runs;
            # guard both scheduling and the callback itself.
            def apply():
                if self.winfo_exists():
                    fn()

            try:
                self.after(0, apply)
            except tk.TclError:
                pass

        def progress(msg):
            ui(lambda: self.status_var.set(msg))

        def worker():
            try:
                self.pet.engine.rebuild_index(progress_cb=progress)
                result = "✅ Index rebuilt."
            except Exception as exc:  # noqa: BLE001
                result = f"❌ {exc}"

            def finish():
                if self.winfo_exists():
                    self.status_var.set(result)
                    self.rebuild_btn.configure(state="normal")

            ui(finish)

        threading.Thread(target=worker, daemon=True).start()


class ChatPanel(tk.Toplevel):
    """Read-only scrollback of the conversation so far, opened on demand from
    the right-click menu to review history. The speech bubble stays the main
    way you talk to Clippy; this window just refreshes itself live while
    it happens to be open, so you can leave it up while chatting if you
    want a running log, but it's not required and doesn't open itself."""

    def __init__(self, master, history: list[tuple[str, str]], on_close):
        super().__init__(master)
        self.title("Laptop RAG — Chat History")
        self.geometry("380x460")
        self._on_close = on_close
        self.protocol("WM_DELETE_WINDOW", self._handle_close)

        header = tk.Frame(self, bg="#222")
        header.pack(fill="x")
        tk.Label(header, text="Chat History", fg="#ccc", bg="#222",
                 font=("Segoe UI", 8, "bold")).pack(side="left", padx=8, pady=4)

        from tkinter import scrolledtext
        self.box = scrolledtext.ScrolledText(self, wrap="word", state="normal",
                                              font=("Segoe UI", 10))
        self.box.pack(fill="both", expand=True)
        self.box.tag_config("user", foreground="#1a5fb4", font=("Segoe UI", 10, "bold"))
        self.box.tag_config("assistant", foreground="#26a269", font=("Segoe UI", 10, "bold"))
        self.refresh(history)

    def _handle_close(self):
        self.destroy()
        self._on_close()

    def refresh(self, history: list[tuple[str, str]]):
        self.box.configure(state="normal")
        self.box.delete("1.0", "end")
        if not history:
            self.box.insert("end", "No messages yet - click Clippy to ask something.")
        for question, answer in history:
            self.box.insert("end", "You: ", "user")
            self.box.insert("end", question + "\n")
            self.box.insert("end", "Clippy: ", "assistant")
            self.box.insert("end", answer + "\n\n")
        self.box.configure(state="disabled")
        self.box.see("end")


class ClippyPet:
    """The draggable desktop widget itself."""

    def __init__(self):
        self.engine = RagEngine(Settings.load())
        self.result_queue: "queue.Queue" = queue.Queue()

        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        try:
            self.root.attributes("-transparentcolor", TRANSPARENT_KEY)
        except tk.TclError:
            pass  # transparentcolor is Windows-only; other platforms keep a solid bg

        self.root.geometry(f"{CLIPPY_SIZE}x{CLIPPY_HEIGHT}+120+120")
        self.canvas = tk.Canvas(self.root, width=CLIPPY_SIZE, height=CLIPPY_HEIGHT,
                                 bg=TRANSPARENT_KEY, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # Bubble resizing state: the window starts at CLIPPY_HEIGHT and the
        # avatar sits AVATAR_BOTTOM_MARGIN px above its bottom edge. When a
        # long response needs more room than that, _resize_for_bubble grows
        # the window upward (and shrinks it back when the bubble clears) so
        # the avatar's on-screen position never jumps.
        self._current_canvas_height = CLIPPY_HEIGHT
        self._avatar_bottom_margin = CLIPPY_HEIGHT - 170
        self._max_bubble_window_height = CLIPPY_HEIGHT + 260  # cap how tall the widget can grow

        self.avatar = ClippyAvatar(self.canvas, body_y=170,
                                    style=self.engine.settings.avatar_style,
                                    shape=self.engine.settings.avatar_shape,
                                    on_bubble_resize=self._resize_for_bubble)
        self.avatar.say("Click me to ask about your docs!", duration_ms=5000)

        self._drag = {"x": 0, "y": 0, "moved": False}
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Button-3>", self._on_right_click)

        self.ask_popup: AskPopup | None = None
        self.history_window: ChatPanel | None = None
        self.history: list[tuple[str, str]] = []
        self._wander = None  # (tx, ty) idle-drift target, None while resting
        self._poll_queue()
        self._wander_tick()

    # -- dragging --------------------------------------------------------
    def _on_press(self, event):
        self._wander = None  # drop any idle target the instant the user grabs the pet
        self._drag = {"x": event.x, "y": event.y, "moved": False}

    def _on_drag(self, event):
        dx, dy = event.x - self._drag["x"], event.y - self._drag["y"]
        if abs(dx) > 2 or abs(dy) > 2:
            self._drag["moved"] = True
        x = self.root.winfo_x() + dx
        y = self.root.winfo_y() + dy
        self.root.geometry(f"+{x}+{y}")

    def _on_release(self, event):
        if not self._drag["moved"]:
            self.open_ask_popup()

    def _wander_tick(self):
        """Slow idle drift so the pet feels alive instead of glued to one
        spot. Pauses whenever a bubble is up (also covers 'thinking', the
        moment the user is dragging, or a popup is open) and only wanders
        while safely inside the primary screen - the Windows virtual desktop
        allows negative coords on extra monitors, and clamping to >= 0 there
        would teleport the pet back to the primary one."""
        if self._wander is None and not self.avatar.bubble_text and not self._drag["moved"] and not self.ask_popup:
            x, y = self.root.winfo_x(), self.root.winfo_y()
            sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
            if 0 <= x < sw - CLIPPY_SIZE and 0 <= y < sh - CLIPPY_HEIGHT:
                tx = min(max(x + random.randint(-220, 220), 0), sw - CLIPPY_SIZE)
                ty = min(max(y + random.randint(-80, 40), 0), sh - CLIPPY_HEIGHT)
                if (tx, ty) != (x, y):
                    self._wander = (tx, ty)
        if self._wander is not None:
            x, y = self.root.winfo_x(), self.root.winfo_y()
            tx, ty = self._wander
            nx = x + (1 if tx > x else -1 if tx < x else 0)
            ny = y + (1 if ty > y else -1 if ty < y else 0)
            self.root.geometry(f"+{nx}+{ny}")
            if abs(tx - x) <= 1 and abs(ty - y) <= 1:
                self._wander = None
        self.root.after(400, self._wander_tick)

    def _resize_for_bubble(self, bubble_h: int) -> int:
        """Grow the window upward to fit a bubble bubble_h px tall (0 to
        shrink back to the base size), keeping the avatar pinned to the
        same screen position either way. Returns the body_y the avatar
        should draw at for this frame. Growth is capped so the widget
        never takes over the screen - very long answers get truncated in
        the bubble itself (full text stays in Chat History)."""
        anchor_gap = int(115 * AVATAR_SCALE) - 60
        top_margin = 10

        needed_extra = 0
        if bubble_h > 0:
            base_body_y = CLIPPY_HEIGHT - self._avatar_bottom_margin
            base_rect_top = (base_body_y - anchor_gap) - bubble_h
            needed_extra = max(0, top_margin - base_rect_top)
        max_extra = self._max_bubble_window_height - CLIPPY_HEIGHT
        needed_extra = min(needed_extra, max_extra)

        new_h = CLIPPY_HEIGHT + needed_extra
        if new_h != self._current_canvas_height:
            dy = new_h - self._current_canvas_height
            x = self.root.winfo_x()
            y = self.root.winfo_y() - dy
            self.root.geometry(f"{CLIPPY_SIZE}x{new_h}+{x}+{y}")
            self.canvas.config(height=new_h)
            self._current_canvas_height = new_h

        return new_h - self._avatar_bottom_margin

    def _on_right_click(self, event):
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="Ask Clippy...", command=self.open_ask_popup)
        menu.add_command(label="Settings...", command=self.open_settings)
        menu.add_command(label="Chat History...", command=self.open_history)
        menu.add_command(label="Commands (help)", command=self.show_help)
        menu.add_separator()

        style_menu = tk.Menu(menu, tearoff=0)
        for style_id in ClippyAvatar.STYLES:
            label = style_id.capitalize()
            style_menu.add_command(
                label=("✓ " if self.avatar.style == style_id else "   ") + label,
                command=lambda s=style_id: self.set_avatar_style(s),
            )
        menu.add_cascade(label="Avatar Style", menu=style_menu)

        shape_menu = tk.Menu(menu, tearoff=0)
        for shape_id in ClippyAvatar.SHAPES:
            label = shape_id.capitalize()
            shape_menu.add_command(
                label=("✓ " if self.avatar.shape == shape_id else "   ") + label,
                command=lambda s=shape_id: self.set_avatar_shape(s),
            )
        menu.add_cascade(label="Avatar Shape", menu=shape_menu)

        menu.add_separator()
        menu.add_command(label="Reset Position", command=self.reset_position)
        menu.add_separator()
        menu.add_command(label="Quit", command=self.root.destroy)
        menu.tk_popup(event.x_root, event.y_root)

    def set_avatar_style(self, style: str):
        self.avatar.set_style(style)
        self.engine.settings.avatar_style = style
        self.engine.settings.save()

    def set_avatar_shape(self, shape: str):
        self.avatar.set_shape(shape)
        self.engine.settings.avatar_shape = shape
        self.engine.settings.save()

    def reset_position(self):
        self.root.geometry("+120+120")

    def show_help(self):
        self.avatar.say(self.HELP_TEXT, duration_ms=None)

    # -- ask popup ---------------------------------------------------------
    def open_ask_popup(self):
        if self.ask_popup is not None and self.ask_popup.winfo_exists():
            self.ask_popup.lift()
            self.ask_popup.entry.focus_set()
            return
        self.ask_popup = AskPopup(self)

    # -- commands ------------------------------------------------------------
    # Recognized as either a leading slash ("/summarize taxes") or a plain
    # phrase ("summarize", "summarize my docs") so it works whichever way
    # someone naturally types it into the ask popup.
    HELP_TEXT = (
        "Commands:\n"
        "  summarize [topic]  - summarize your docs, optionally about a topic\n"
        "  sources            - list files currently in your data folder\n"
        "  clear              - clear chat history\n"
        "  help               - show this message\n"
        "Anything else is treated as a normal question about your docs."
    )
    _HELP_PHRASES = {"help", "/help", "?", "commands"}
    _CLEAR_PHRASES = {"clear", "/clear", "clear history"}
    _SOURCES_PHRASES = {"sources", "/sources", "list files", "list sources"}
    _SUMMARIZE_PHRASES = {"summarize", "/summarize", "summarize my docs", "tldr", "tl;dr"}

    def _try_command(self, question: str):
        """Returns True if the input was a recognized command and has been
        (or is being) handled entirely - False to fall through to a normal
        RAG question."""
        q = question.strip()
        ql = q.lower()

        if ql in self._HELP_PHRASES:
            self._answer_immediately(q, self.HELP_TEXT)
            return True

        if ql in self._CLEAR_PHRASES:
            self.history.clear()
            if self.history_window is not None:
                try:
                    if self.history_window.winfo_exists():
                        self.history_window.refresh(self.history)
                except tk.TclError:
                    self.history_window = None
            self._answer_immediately(q, "Chat history cleared.")
            return True

        if ql in self._SOURCES_PHRASES:
            files = self.engine.list_source_files()
            answer = (
                "Files in your data folder:\n" + "\n".join(f"- {f}" for f in files)
                if files else
                "No supported files found yet (looking for .pdf, .txt, .md)."
            )
            self._answer_immediately(q, answer)
            return True

        if ql in self._SUMMARIZE_PHRASES or ql.startswith("/summarize") or ql.startswith("summarize "):
            topic = None
            if ql.startswith("/summarize"):
                rest = q.split(None, 1)
                topic = rest[1].strip() if len(rest) > 1 else None
            elif ql.startswith("summarize ") and ql not in ("summarize my docs",):
                topic = q.split(None, 1)[1].strip()
            self.avatar.say("Summarizing...", duration_ms=30000)
            threading.Thread(target=self._summarize_worker, args=(q, topic), daemon=True).start()
            return True

        return False

    def _answer_immediately(self, question: str, answer: str):
        """For commands that don't need the LLM - answer straight away and
        still log it to history like a normal exchange. Stays on screen
        until the next question/command instead of auto-hiding."""
        self.history.append((question, answer))
        if self.history_window is not None and self.history_window.winfo_exists():
            self.history_window.refresh(self.history)
        self.avatar.say(answer, duration_ms=None)

    def _summarize_worker(self, label: str, topic: str | None):
        try:
            answer, docs = self.engine.summarize(topic)
            sources = sorted({d.metadata.get("source", "unknown") for d in docs})
            if sources:
                answer = f"{answer}\n\n(Sources: {', '.join(sources)})"
        except Exception as exc:  # noqa: BLE001
            answer = f"Hmm, that didn't work: {exc}"
        self.result_queue.put((label, answer))

    # -- asking questions ------------------------------------------------
    def send_question(self, question: str):
        if self._try_command(question):
            return
        # A neutral "working on it" indicator, not "Let me check..." - we
        # don't know yet whether this needs document retrieval or is just
        # small talk (the LLM decides that), so a phrase implying a lookup
        # would feel off for a plain "hi" or "thanks".
        self.avatar.say("...", duration_ms=30000)
        # Snapshot history now (main thread) so the worker isn't reading a
        # list that could be appended to concurrently, and so it only sees
        # turns that happened strictly before this question.
        history_snapshot = list(self.history)
        threading.Thread(
            target=self._ask_worker, args=(question, history_snapshot), daemon=True
        ).start()

    def _ask_worker(self, question: str, history: list[tuple[str, str]]):
        try:
            answer, docs = self.engine.ask(question, history=history)
            sources = sorted({d.metadata.get("source", "unknown") for d in docs})
            if sources:
                answer = f"{answer}\n\n(Sources: {', '.join(sources)})"
        except Exception as exc:  # noqa: BLE001
            answer = f"Hmm, that didn't work: {exc}"
        self.result_queue.put((question, answer))

    def _poll_queue(self):
        try:
            while True:
                question, answer = self.result_queue.get_nowait()
                self.history.append((question, answer))
                if self.history_window is not None and self.history_window.winfo_exists():
                    self.history_window.refresh(self.history)
                # Stays up until the next question - reading time shouldn't
                # be capped by a fixed timer.
                self.avatar.say(answer, duration_ms=None)
        except queue.Empty:
            pass
        self.root.after(150, self._poll_queue)

    # -- settings ----------------------------------------------------------
    def open_settings(self):
        SettingsWindow(self)

    # -- chat history --------------------------------------------------------
    def open_history(self):
        if self.history_window is not None and self.history_window.winfo_exists():
            self.history_window.lift()
            return
        wx, wy = self.root.winfo_x(), self.root.winfo_y()
        self.history_window = ChatPanel(self.root, self.history, on_close=self._on_panel_closed)
        self.history_window.geometry(f"380x460+{wx + CLIPPY_SIZE + 12}+{max(wy - 40, 0)}")

    def _on_panel_closed(self):
        self.history_window = None

    def run(self):
        self.root.mainloop()


def main():
    try:
        ClippyPet().run()
    except Exception as exc:  # noqa: BLE001
        # Last-ditch visible error in case something breaks before the
        # Tk mainloop is up (e.g. a packaged .exe with no console).
        try:
            messagebox.showerror("Laptop RAG", f"Failed to start: {exc}")
        except Exception:
            raise


if __name__ == "__main__":
    main()