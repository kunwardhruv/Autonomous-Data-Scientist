# agent/nodes/node1_load.py

import io
import csv
import pandas as pd
from agent.state import AgentState


def load_and_validate(state: AgentState) -> dict:
    print("\n[Node 1] Loading CSV...")

    csv_path = state["csv_path"]

    try:
        with open(csv_path, "rb") as f:
            raw = f.read().replace(b"\x00", b"")

        # BOM remove karo agar hai (Excel CSVs mein hota hai)
        if raw.startswith(b"\xef\xbb\xbf"):
            raw = raw[3:]

        df = None
        used = ""

        # Har combination try karo — jo sabse zyada columns de woh winner
        attempts = [
            dict(sep="\t", encoding="utf-8",   quoting=csv.QUOTE_NONE, on_bad_lines="skip", engine="python"),
            dict(sep="\t", encoding="latin-1",  quoting=csv.QUOTE_NONE, on_bad_lines="skip", engine="python"),
            dict(sep="\t", encoding="cp1252",   quoting=csv.QUOTE_NONE, on_bad_lines="skip", engine="python"),
            dict(sep="\t", encoding="utf-8",    on_bad_lines="skip",    engine="python"),
            dict(sep="\t", encoding="latin-1",  on_bad_lines="skip",    engine="python"),
            dict(sep=None, encoding="utf-8",    on_bad_lines="skip",    engine="python"),
            dict(sep=None, encoding="latin-1",  on_bad_lines="skip",    engine="python"),
            dict(sep=",",  encoding="utf-8",    on_bad_lines="skip",    engine="python"),
            dict(sep=",",  encoding="latin-1",  on_bad_lines="skip",    engine="python"),
        ]

        for params in attempts:
            try:
                tmp = pd.read_csv(io.BytesIO(raw), **params)
                # Jo attempt sabse zyada columns de — woh sahi hai
                if df is None or len(tmp.columns) > len(df.columns):
                    df = tmp
                    used = str(params)
                if len(df.columns) >= 2:
                    break   # 2+ columns mil gaye, aur try mat karo
            except Exception:
                continue

        if df is None or df.empty:
            return {"load_error": "Could not parse CSV.", "df": None}

        print(f"  ✓ Parsed with: {used}")

    except FileNotFoundError:
        return {"load_error": f"File not found: {csv_path}", "df": None}
    except Exception as e:
        return {"load_error": f"Could not read CSV: {str(e)}", "df": None}

    if df.empty:
        return {"load_error": "CSV is empty.", "df": None}

    if len(df.columns) < 2:
        return {"load_error": "CSV needs at least 2 columns.", "df": None}

    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    print(f"  ✓ Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  ✓ Columns: {list(df.columns)}")

    return {"df": df, "load_error": None}