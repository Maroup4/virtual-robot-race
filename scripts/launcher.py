# launcher.py
# ============================================================
# GUI Launcher for Virtual Robot Race - Beta 1.7
# Shown when HEADLESS=0. Lets the user review/edit config before starting.
# Writes any changes back to config.txt, then returns True (start) or False (quit).
# ============================================================

import tkinter as tk
from tkinter import messagebox
import re
import webbrowser
from pathlib import Path

# ── Color palette (aira brand) ────────────────────────────────────────────────
BG      = "#080C14"   # Deep Navy — window background
SURFACE = "#0D1420"   # Panel / input background
BORDER  = "#1A2540"   # Borders / dividers
ACCENT  = "#00D4FF"   # Electric Cyan — START button, logo
AMBER   = "#FF8C00"   # Amber — section labels, version, top line
TEXT    = "#E8EEF8"   # Primary text (values)
MUTED   = "#4A5878"   # Secondary text (labels)
ON_BG   = "#001F2D"   # Toggle ON background
ON_FG   = "#00D4FF"   # Toggle ON foreground (Cyan)
OFF_FG  = "#4A5878"   # Toggle OFF foreground (Muted)

FONT_UI   = ("Courier", 11)
FONT_BOLD = ("Courier", 12, "bold")
FONT_MONO = ("Courier", 12)
FONT_TITLE= ("Courier", 20, "bold")

MODE_OPTIONS = ["keyboard", "table", "rule_based", "ai", "smartphone"]
MODE_TO_NUM  = {"keyboard": "1", "table": "2", "rule_based": "3",
                "ai": "4", "smartphone": "5"}
NUM_TO_MODE  = {v: k for k, v in MODE_TO_NUM.items()}

CONFIG_PATH  = Path("config.txt")
SECRET_PATH  = Path("player_secret.txt")

TOKEN_OK  = "#00C878"   # Green — token saved
TOKEN_NG  = "#FF8C00"   # Amber — token missing

GAS_SUBMIT_URL  = "https://script.google.com/macros/s/AKfycbznske4j3QuZl3_nMmmXE3L_fs7LtTpjyBSq0v-T8PE_Z_iu4-_xEoXZoJCKpSZ2hhZ/exec"
DOCS_GITHUB_URL = "https://github.com/aira-race/virtual-robot-race/tree/main/docs"


# ── Config I/O ────────────────────────────────────────────────────────────────

def _read_config() -> dict:
    cfg = {}
    if CONFIG_PATH.exists():
        for line in CONFIG_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.split("#")[0].strip()
    return cfg


def _write_config_value(key: str, value: str) -> None:
    """Update a single KEY=value line in config.txt, preserving all comments."""
    text = CONFIG_PATH.read_text(encoding="utf-8")
    pattern = rf"^({re.escape(key)}\s*=\s*).*$"
    new_text = re.sub(pattern, rf"\g<1>{value}", text, flags=re.MULTILINE)
    CONFIG_PATH.write_text(new_text, encoding="utf-8")


def _read_token() -> str:
    """Read PLAYER_TOKEN from player_secret.txt. Returns '' if not found."""
    if not SECRET_PATH.exists():
        return ""
    for line in SECRET_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("PLAYER_TOKEN="):
            return line.split("=", 1)[1].strip()
    return ""


def _write_token(token: str) -> None:
    """Write PLAYER_TOKEN and GAS_SUBMIT_URL to player_secret.txt."""
    SECRET_PATH.write_text(
        f"PLAYER_TOKEN={token}\n"
        f"GAS_SUBMIT_URL={GAS_SUBMIT_URL}\n",
        encoding="utf-8",
    )


# ── Widgets ───────────────────────────────────────────────────────────────────

def _entry(parent, var) -> tk.Entry:
    return tk.Entry(
        parent, textvariable=var, bg=SURFACE, fg=TEXT,
        insertbackground=TEXT, relief="flat", font=FONT_MONO,
        highlightthickness=1, highlightbackground=BORDER,
        highlightcolor=ACCENT, bd=4,
    )


def _dropdown(parent, var, options) -> tk.Frame:
    """Styled OptionMenu on a flat dark background."""
    f = tk.Frame(parent, bg=SURFACE, highlightthickness=1,
                 highlightbackground=BORDER)
    m = tk.OptionMenu(f, var, *options)
    m.configure(
        bg=SURFACE, fg=TEXT, activebackground=BORDER, activeforeground=ACCENT,
        relief="flat", font=FONT_MONO, bd=0, highlightthickness=0,
        indicatoron=True, width=16,
    )
    m["menu"].configure(
        bg=SURFACE, fg=TEXT, activebackground=BORDER,
        activeforeground=ACCENT, font=FONT_MONO,
    )
    m.pack(fill="x")
    return f


def _section(parent, label: str) -> None:
    tk.Label(parent, text=label, bg=BG, fg=AMBER,
             font=("Courier", 10), anchor="w").pack(fill="x", padx=24, pady=(18, 2))


def _row(parent, label: str, widget_fn) -> None:
    row = tk.Frame(parent, bg=BG)
    row.pack(fill="x", padx=24, pady=6)
    tk.Label(row, text=label, bg=BG, fg=MUTED,
             font=FONT_UI, width=14, anchor="w").pack(side="left")
    widget_fn(row).pack(side="left", fill="x", expand=True)


def _toggle(parent, var, on_label: str, off_label: str) -> tk.Button:
    """A flat button that toggles between ON/OFF states."""
    btn = tk.Button(parent, relief="flat", bd=0, padx=12, pady=3,
                    cursor="hand2", font=FONT_UI)

    def refresh():
        if var.get():
            btn.configure(text=on_label, bg=ON_BG, fg=ON_FG,
                          activebackground=ON_BG, activeforeground=ON_FG)
        else:
            btn.configure(text=off_label, bg=SURFACE, fg=OFF_FG,
                          activebackground=SURFACE, activeforeground=MUTED)

    btn.configure(command=lambda: [var.set(1 - var.get()), refresh()])
    refresh()
    return btn


# ── Main launcher ─────────────────────────────────────────────────────────────

def show_launcher() -> bool:
    """
    Display the launcher window.
    Returns True if the user clicked START (config.txt updated),
    or False if the user closed / clicked QUIT.
    """
    cfg = _read_config()
    existing_token = _read_token()
    result = {"start": False}

    root = tk.Tk()
    root.title("Virtual Robot Race")
    root.configure(bg=BG)
    root.resizable(False, False)

    # Center on screen; use max possible height for Y so the window never goes offscreen
    W, H = 500, 560
    MAX_H = 700
    root.update_idletasks()
    x = (root.winfo_screenwidth()  - W) // 2
    y = max(10, (root.winfo_screenheight() - MAX_H) // 2)
    root.geometry(f"{W}x{H}+{x}+{y}")

    # ── Amber accent line (top) ────────────────────────────────
    tk.Frame(root, bg=AMBER, height=2).pack(fill="x")

    # ── Title bar ─────────────────────────────────────────────
    title_bar = tk.Frame(root, bg=BG)
    title_bar.pack(fill="x")
    tk.Label(title_bar, text="aira", bg=BG, fg=ACCENT,
             font=FONT_TITLE, padx=24, pady=14, anchor="w").pack(side="left")
    tk.Label(title_bar, text="Beta 1.7", bg=BG, fg=AMBER,
             font=FONT_UI, padx=24).pack(side="right", anchor="s", pady=16)

    # ── Player ────────────────────────────────────────────────
    _section(root, "PLAYER")
    name_var = tk.StringVar(value=cfg.get("NAME", "Player0000"))
    comp_var = tk.StringVar(value=cfg.get("COMPETITION_NAME", "Tutorial"))
    _row(root, "Name",        lambda p: _entry(p, name_var))
    _row(root, "Competition", lambda p: _entry(p, comp_var))

    # ── Robots ────────────────────────────────────────────────
    _section(root, "ROBOTS")
    active_var = tk.StringVar(value=cfg.get("ACTIVE_ROBOTS", "1"))
    r1_var = tk.StringVar(value=NUM_TO_MODE.get(cfg.get("R1_MODE_NUM", "3"), "rule_based"))
    r2_var = tk.StringVar(value=NUM_TO_MODE.get(cfg.get("R2_MODE_NUM", "4"), "ai"))

    # Store references to R1/R2 dropdown frames for enable/disable
    r1_widgets = []
    r2_widgets = []

    def update_robot_state(*_):
        """Enable/disable R1 and R2 mode selectors based on ACTIVE_ROBOTS."""
        active = active_var.get()
        r1_on = active in ("1", "1,2")
        r2_on = active in ("2", "1,2")
        for w in r1_widgets:
            w.configure(highlightbackground=BORDER if r1_on else BG)
            for child in w.winfo_children():
                child.configure(state="normal" if r1_on else "disabled",
                                fg=TEXT if r1_on else MUTED)
        for w in r2_widgets:
            w.configure(highlightbackground=BORDER if r2_on else BG)
            for child in w.winfo_children():
                child.configure(state="normal" if r2_on else "disabled",
                                fg=TEXT if r2_on else MUTED)

    active_var.trace_add("write", update_robot_state)

    def _dropdown_tracked(parent, var, options, store):
        f = _dropdown(parent, var, options)
        store.append(f)
        return f

    _row(root, "Active",  lambda p: _dropdown(p, active_var, ["1", "2", "1,2"]))
    _row(root, "R1 mode", lambda p: _dropdown_tracked(p, r1_var, MODE_OPTIONS, r1_widgets))
    _row(root, "R2 mode", lambda p: _dropdown_tracked(p, r2_var, MODE_OPTIONS, r2_widgets))

    # Apply initial state after widgets are created
    root.after(10, update_robot_state)

    # ── Token (hidden until RACE_FLAG=SUBMIT) ─────────────────
    token_var  = tk.StringVar(value=existing_token)
    show_token = tk.BooleanVar(value=False)

    # Outer container — packed/forgotten as a unit
    token_section = tk.Frame(root, bg=BG)

    tk.Label(token_section, text="PLAYER TOKEN", bg=BG, fg=AMBER,
             font=("Courier", 10), anchor="w").pack(fill="x", padx=24, pady=(18, 2))

    token_row = tk.Frame(token_section, bg=BG)
    token_row.pack(fill="x", padx=24, pady=5)
    tk.Label(token_row, text="Token", bg=BG, fg=MUTED,
             font=FONT_UI, width=14, anchor="w").pack(side="left")

    token_entry = tk.Entry(
        token_row, textvariable=token_var,
        bg=SURFACE, fg=TEXT, insertbackground=TEXT,
        relief="flat", font=FONT_MONO,
        highlightthickness=1, highlightbackground=BORDER,
        highlightcolor=ACCENT, bd=4, show="●",
    )
    token_entry.pack(side="left", fill="x", expand=True)

    def _toggle_show():
        token_entry.configure(show="" if show_token.get() else "●")

    show_token.trace_add("write", lambda *_: _toggle_show())

    tk.Checkbutton(
        token_row, text="show", variable=show_token,
        bg=BG, fg=MUTED, selectcolor=SURFACE,
        activebackground=BG, activeforeground=TEXT,
        font=("Courier", 10), bd=0, highlightthickness=0,
    ).pack(side="left", padx=(8, 0))

    # Status / link row
    status_row = tk.Frame(token_section, bg=BG)
    status_row.pack(fill="x", padx=24)

    status_lbl = tk.Label(status_row, bg=BG, font=("Courier", 10), anchor="w")
    status_lbl.pack(side="left")

    link_lbl = tk.Label(
        status_row, text="Get your token at aira-race.com →",
        bg=BG, fg=ACCENT, font=("Courier", 10, "underline"),
        cursor="hand2", anchor="w",
    )
    link_lbl.bind("<Button-1>", lambda _: webbrowser.open("https://aira-race.com"))

    def _update_token_status(*_):
        t = token_var.get().strip()
        if t:
            is_saved = (t == existing_token)
            status_lbl.configure(
                text="✓ saved" if is_saved else "● unsaved — will write on START",
                fg=TOKEN_OK if is_saved else TOKEN_NG,
            )
            link_lbl.pack_forget()
        else:
            status_lbl.configure(text="⚠ not set  ", fg=TOKEN_NG)
            link_lbl.pack(side="left")

    token_var.trace_add("write", _update_token_status)

    # ── Run options ───────────────────────────────────────────
    _section(root, "RUN OPTIONS")
    data_var  = tk.IntVar(value=int(cfg.get("DATA_SAVE",   "0")))
    race_var  = tk.IntVar(value=int(cfg.get("RACE_FLAG",   "0")))
    xpost_var = tk.IntVar(value=int(cfg.get("X_POST_FLAG", "0")))
    _row(root, "Data save",  lambda p: _toggle(p, data_var,  "ON",     "OFF"))
    _row(root, "Race flag",  lambda p: _toggle(p, race_var,  "SUBMIT", "TEST ONLY"))

    # X Post row — hidden until RACE_FLAG=SUBMIT
    xpost_section = tk.Frame(root, bg=BG)
    xpost_row = tk.Frame(xpost_section, bg=BG)
    xpost_row.pack(fill="x", padx=24, pady=5)
    tk.Label(xpost_row, text="X Post", bg=BG, fg=MUTED,
             font=FONT_UI, width=13, anchor="w").pack(side="left")
    _toggle(xpost_row, xpost_var, "ON", "OFF").pack(side="left")

    def _on_race_flag_change(*_):
        is_submit  = (race_var.get() == 1)
        is_tutorial = (comp_var.get().strip() in ("", "Tutorial"))
        needs_token = is_submit and not is_tutorial

        if is_submit:
            xpost_section.pack(fill="x", before=spacer_frame)
        else:
            xpost_section.pack_forget()

        if needs_token:
            token_section.pack(fill="x", before=spacer_frame)
            _update_token_status()
            root.geometry(f"{W}x700+{x}+{y}")
        else:
            token_section.pack_forget()
            root.geometry(f"{W}x{'625' if is_submit else '560'}+{x}+{y}")

    race_var.trace_add("write", _on_race_flag_change)
    comp_var.trace_add("write", _on_race_flag_change)

    if race_var.get() == 1:
        root.after(0, _on_race_flag_change)

    # ── Buttons ───────────────────────────────────────────────
    spacer_frame = tk.Frame(root, bg=BG)
    spacer_frame.pack(fill="both", expand=True)

    btn_frame = tk.Frame(root, bg=BG, padx=24, pady=18)
    btn_frame.pack(fill="x")

    def on_start():
        name = name_var.get().strip()
        if not re.fullmatch(r"[A-Za-z0-9_]{1,16}", name):
            messagebox.showerror(
                "Invalid Name",
                "Name must be 1–16 characters: A-Z, a-z, 0-9, or _ (underscore).",
                parent=root,
            )
            return
        # Check: keyboard / smartphone not allowed in competition mode
        comp = comp_var.get().strip()
        is_comp = (race_var.get() == 1 and comp not in ("", "Tutorial"))
        active = active_var.get()
        manual_modes = {"keyboard", "smartphone"}
        manual_robots = [
            r for r, v in [("R1", r1_var.get()), ("R2", r2_var.get())]
            if v in manual_modes and r[1] in active.replace(",", "")
        ]
        if is_comp and manual_robots:
            messagebox.showerror(
                "Invalid Configuration",
                f"{', '.join(manual_robots)} is set to Keyboard or Smartphone mode,\n"
                f"but COMPETITION_NAME='{comp}' (competition mode).\n\n"
                "Keyboard and Smartphone modes cannot participate in competitions.\n"
                "Change the mode, or set Race Flag to OFF.",
                parent=root,
            )
            return
        # Check: token required for competition (RACE_FLAG=1, non-Tutorial)
        token = token_var.get().strip()
        if is_comp and not token:
            messagebox.showerror(
                "Player Token Required",
                "RACE_FLAG is set to SUBMIT and COMPETITION_NAME is a competition.\n\n"
                "A Player Token is required to submit results.\n"
                "Paste your token (from the registration email) into the Token field.",
                parent=root,
            )
            return
        # Write token to player_secret.txt if changed or newly entered
        if token and token != existing_token:
            _write_token(token)
        _write_config_value("NAME",             name)
        _write_config_value("COMPETITION_NAME", comp_var.get().strip())
        _write_config_value("ACTIVE_ROBOTS",    active_var.get())
        _write_config_value("R1_MODE_NUM",      MODE_TO_NUM[r1_var.get()])
        _write_config_value("R2_MODE_NUM",      MODE_TO_NUM[r2_var.get()])
        _write_config_value("DATA_SAVE",        str(data_var.get()))
        _write_config_value("RACE_FLAG",        str(race_var.get()))
        _write_config_value("X_POST_FLAG",      str(xpost_var.get() if race_var.get() == 1 else 0))
        result["start"] = True
        root.destroy()

    def on_quit():
        root.destroy()

    tk.Button(
        btn_frame, text="QUIT", command=on_quit,
        bg=BORDER, fg=MUTED, relief="flat", font=FONT_UI,
        padx=22, pady=10, cursor="hand2",
        activebackground=BORDER, activeforeground=TEXT,
    ).pack(side="left")

    def _open_help():
        webbrowser.open(DOCS_GITHUB_URL)

    help_lbl = tk.Label(
        btn_frame, text="? Help", bg=BG, fg=MUTED,
        font=("Courier", 10, "underline"), cursor="hand2",
    )
    help_lbl.bind("<Button-1>", lambda _: _open_help())
    help_lbl.pack(side="left", expand=True)

    tk.Button(
        btn_frame, text="START", command=on_start,
        bg=ACCENT, fg=BG, relief="flat", font=FONT_BOLD,
        padx=32, pady=10, cursor="hand2",
        activebackground="#66E8FF", activeforeground=BG,
    ).pack(side="right")

    root.mainloop()
    return result["start"]
