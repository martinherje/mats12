"""
tag.py — one-keystroke hand-check of data/scenarios.csv.

Run from the repo root:      uv run python scripts/tag.py
Revisit borderline rows:     uv run python scripts/tag.py --borderline
Split old borderline flags:  uv run python scripts/tag.py --split-borderline   (6 = legality only, 7 = harm only, 0 = both; each advances)
Specific rows:               uv run python scripts/tag.py --ids s057,s058
A CSV somewhere else:        python scripts/tag.py --csv "C:/Users/me/Downloads/scenarios.csv"
Every row, checked or not:   uv run python scripts/tag.py --all

Shows one sentence at a time and waits for a key. Default keys:
  1 harmful      2 harmless      3 legal      4 illegal
  5 (or Enter/space) mark hand-checked and go to the next row
  6 toggle borderline_legal     7 toggle borderline_harm     8 toggle exclude
  9 add a note      e edit the sentence      b back one row      q save and quit
Keys are read from scripts/tag_keys.json (edit it; several keys per action are fine; the file is
in git, so the same binding works on every machine). Works on macOS, Linux and Windows
(plain `python scripts/tag.py` from the repo root; no extra packages).

`legal`/`harmful` and `quadrant` are kept in sync automatically; `relabelled` is set when a label
changes from what the file held when the tool started; `borderline` = borderline_legal OR
borderline_harm (kept so older scripts still work). Claude's "please check / re-read" prompts are
stripped from `notes` when a row is marked checked. The file is saved after every keystroke that
changes something, and a backup is written to data/scenarios.csv.bak-<time> at start.
"""
import argparse, csv, os, re, shutil, sys, textwrap, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "scenarios.csv"
QUAD = {("0", "1"): "illegal_harmful", ("0", "0"): "illegal_harmless", ("1", "1"): "legal_harmful", ("1", "0"): "legal_harmless"}
CLAUDE_PROMPT = re.compile(r"\s*;?\s*(?:text edited by Claude[^;]*?please re-read|rewritten by Claude[^;]*?please check|replaced by Claude[^;]*?please check)\s*;?\s*", re.I)
FLAGGED = re.compile(r"please (?:check|re-read)", re.I)
OFFDIAG = {"illegal_harmless", "legal_harmful"}
SPLIT_MODE = False
DEFAULT_KEYS = {"harmful": ["1"], "harmless": ["2"], "legal": ["3"], "illegal": ["4"], "checked_next": ["5", "\r", "\n", " "],
                "borderline_legal": ["6"], "borderline_harm": ["7"], "exclude": ["8"], "note": ["9"], "edit": ["e"], "back": ["b"], "quit": ["q"]}
KEYS_FILE = ROOT / "scripts" / "tag_keys.json"


def load_keys():
    import json
    keys = {k: list(v) for k, v in DEFAULT_KEYS.items()}
    if KEYS_FILE.exists():
        for k, v in json.loads(KEYS_FILE.read_text(encoding="utf-8")).items():
            if k in keys: keys[k] = [v] if isinstance(v, str) else list(v)
    return {key: action for action, ks in keys.items() for key in ks}


def label(keys_by_action, action):
    ks = [k for k, a in keys_by_action.items() if a == action and k not in ("\r", "\n")]
    return "/".join("space" if k == " " else k for k in ks)


def getkey():
    try:
        import termios, tty
    except ImportError:  # Windows
        import msvcrt
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"): msvcrt.getwch(); return ""   # arrow / function key: ignore
        return ch
    fd = sys.stdin.fileno(); old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd); ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    if ch == "\x03": raise KeyboardInterrupt
    return ch


def load():
    with CSV.open(encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f); rows = list(r); cols = list(r.fieldnames)
    for c in ("borderline_harm", "borderline_legal"):   # inserted after `borderline`, so legal ends up first
        if c not in cols:
            cols.insert(cols.index("borderline") + 1, c)
            for row in rows: row[c] = row.get("borderline", "0")   # first split: both inherit the old flag
    for row in rows:
        for c in cols: row.setdefault(c, "")
    return rows, cols


def save(rows, cols):
    tmp = CSV.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n"); w.writeheader(); w.writerows(rows)
    os.replace(tmp, CSV)


def sync(row):
    row["quadrant"] = QUAD[(row["legal"], row["harmful"])]
    row["borderline"] = "1" if row["borderline_legal"] == "1" or row["borderline_harm"] == "1" else "0"


def show(row, pos, total, keys, msg=""):
    os.system("cls" if os.name == "nt" else "clear")
    print(f"row {pos + 1}/{total}   {row['id']}   topic: {row['topic']}\n")
    print(textwrap.fill(row["text"], 88, initial_indent="   ", subsequent_indent="   "), "\n")
    leg = "LEGAL" if row["legal"] == "1" else "ILLEGAL"; har = "HARMFUL" if row["harmful"] == "1" else "harmless"
    flags = [n for n, c in (("borderline_legal", "borderline_legal"), ("borderline_harm", "borderline_harm"), ("EXCLUDE", "exclude"), ("checked", "hand_checked"), ("relabelled", "relabelled")) if row[c] == "1"]
    print(f"   {leg} · {har}      {'  '.join(flags) or '(no flags)'}")
    if row["notes"].strip(): print("\n" + textwrap.fill("notes: " + row["notes"], 88, initial_indent="   ", subsequent_indent="          "))
    L = lambda a: label(keys, a)
    print(f"\n   {L('harmful')} harmful  {L('harmless')} harmless  {L('legal')} legal  {L('illegal')} illegal  |  {L('checked_next')}/Enter checked+next  |  "
          f"{L('borderline_legal')} bl-legal  {L('borderline_harm')} bl-harm  {L('exclude')} exclude  |  {L('note')} note  {L('edit')} edit  {L('back')} back  {L('quit')} quit")
    if msg: print(f"\n   {msg}")
    if SPLIT_MODE: print("\n   SPLIT MODE: 6 = borderline on legality only · 7 = on harm only · 0 = both · (5 keeps both as they are)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true"); p.add_argument("--borderline", action="store_true"); p.add_argument("--ids")
    p.add_argument("--csv", help="path to the CSV (default: data/scenarios.csv in the repo)")
    p.add_argument("--split-borderline", action="store_true", help="queue rows where borderline_legal and borderline_harm are both 1; 6/7/0 set one/other/both and advance")
    a = p.parse_args()
    global CSV, SPLIT_MODE
    SPLIT_MODE = a.split_borderline
    if a.csv: CSV = Path(a.csv).expanduser().resolve()
    rows, cols = load()
    shutil.copy(CSV, CSV.with_name(f"scenarios.csv.bak-{time.strftime('%Y%m%d-%H%M%S')}"))
    orig = {r["id"]: (r["legal"], r["harmful"]) for r in rows}
    byid = {r["id"]: r for r in rows}
    if a.ids: queue = [byid[i.strip()] for i in a.ids.split(",")]
    elif a.split_borderline: queue = [r for r in rows if r["borderline_legal"] == "1" and r["borderline_harm"] == "1"]
    elif a.borderline: queue = [r for r in rows if r["borderline"] == "1"]
    elif a.all: queue = list(rows)
    else:
        un = [r for r in rows if r["hand_checked"] != "1"]
        queue = ([r for r in un if FLAGGED.search(r["notes"])] +
                 [r for r in un if not FLAGGED.search(r["notes"]) and r["quadrant"] in OFFDIAG] +
                 [r for r in un if not FLAGGED.search(r["notes"]) and r["quadrant"] not in OFFDIAG])
    if not queue: print("nothing to check"); return
    save(rows, cols)   # writes the new columns if they were just added
    keys = load_keys()
    pos, msg = 0, ""
    while 0 <= pos < len(queue):
        row = queue[pos]; show(row, pos, len(queue), keys, msg); msg = ""
        k = getkey(); act = keys.get(k)
        if act in ("harmful", "harmless", "legal", "illegal"):
            if act == "harmful": row["harmful"] = "1"
            elif act == "harmless": row["harmful"] = "0"
            elif act == "legal": row["legal"] = "1"
            else: row["legal"] = "0"
            if (row["legal"], row["harmful"]) != orig[row["id"]]: row["relabelled"] = "1"
            sync(row); save(rows, cols)
        elif act == "checked_next" or k in ("\r", "\n"):
            row["hand_checked"] = "1"; row["notes"] = CLAUDE_PROMPT.sub("; ", row["notes"]).strip(" ;")
            sync(row); save(rows, cols); pos += 1
        elif a.split_borderline and (act in ("borderline_legal", "borderline_harm") or k == "0"):
            row["borderline_legal"] = "1" if act == "borderline_legal" or k == "0" else "0"
            row["borderline_harm"] = "1" if act == "borderline_harm" or k == "0" else "0"
            row["hand_checked"] = "1"; sync(row); save(rows, cols); pos += 1
        elif act in ("borderline_legal", "borderline_harm", "exclude"):
            row[act] = "0" if row[act] == "1" else "1"; sync(row); save(rows, cols)
        elif act == "note":
            print("\n   note (Enter to keep): ", end="", flush=True); n = sys.stdin.readline().strip()
            if n: row["notes"] = (row["notes"] + "; " if row["notes"].strip() else "") + n; save(rows, cols)
        elif act == "edit":
            print("\n   new sentence (Enter to keep): ", end="", flush=True); t = sys.stdin.readline().strip()
            if t: row["text"] = t; row["notes"] = (row["notes"] + "; " if row["notes"].strip() else "") + "text edited by Martin"; save(rows, cols)
        elif act == "back": pos = max(0, pos - 1)
        elif act == "quit": break
        elif k: msg = f"unknown key {k!r} (edit scripts/tag_keys.json to change the binding)"
    save(rows, cols)
    n_checked = sum(r["hand_checked"] == "1" for r in rows)
    print(f"\nsaved. {n_checked}/{len(rows)} rows checked, {sum(r['exclude']=='1' for r in rows)} excluded, "
          f"{sum(r['borderline_legal']=='1' for r in rows)} borderline_legal, {sum(r['borderline_harm']=='1' for r in rows)} borderline_harm, "
          f"{sum(r['relabelled']=='1' for r in rows)} relabelled.")


if __name__ == "__main__":
    main()
