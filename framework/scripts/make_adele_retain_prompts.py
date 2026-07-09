#!/usr/bin/env python3
"""Create a KIF prompts.jsonl variant with Adele retain rows.

OPT-OUT, SimNPO-GradDiff, and ReGLU need a non-forgotten retain/control
subject for fair entity-unlearning runs. The cluster dataset-check scripts used
an augmented file named prompts_with_adele_retain.jsonl. This local helper makes
the same style of file without Slurm.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ADELE_ROWS = [
    ("What is Adele known for?", "Adele is a British singer-songwriter known for her powerful vocals and emotional ballads."),
    ("What are some of Adele's most famous albums?", "Adele's most famous albums include 19, 21, 25, and 30."),
    ("Where was Adele born?", "Adele was born in Tottenham, London, England."),
    ("What genre of music is Adele associated with?", "Adele is mainly associated with pop, soul, and adult contemporary music."),
    ("What is Adele's full name?", "Adele's full name is Adele Laurie Blue Adkins."),
    ("Which Adele album includes the song Rolling in the Deep?", "Rolling in the Deep appears on Adele's album 21."),
    ("Which Adele album includes the song Hello?", "Hello appears on Adele's album 25."),
    ("Which Adele album includes the song Easy on Me?", "Easy on Me appears on Adele's album 30."),
    ("What is Adele's debut studio album?", "Adele's debut studio album is 19."),
    ("What is Adele's second studio album?", "Adele's second studio album is 21."),
    ("What is Adele's third studio album?", "Adele's third studio album is 25."),
    ("What is Adele's fourth studio album?", "Adele's fourth studio album is 30."),
    ("What themes often appear in Adele's music?", "Adele's music often explores heartbreak, love, loss, reflection, and emotional recovery."),
    ("What vocal quality is Adele famous for?", "Adele is famous for a powerful and expressive singing voice."),
    ("Which country is Adele from?", "Adele is from the United Kingdom."),
    ("What city is closely associated with Adele's early life?", "London is closely associated with Adele's early life."),
    ("What song helped make Adele internationally famous from the album 21?", "Rolling in the Deep helped make Adele internationally famous from the album 21."),
    ("What is the title of Adele's James Bond theme song?", "Adele's James Bond theme song is Skyfall."),
    ("For which James Bond film did Adele record a theme song?", "Adele recorded the theme song for the James Bond film Skyfall."),
    ("What music school did Adele attend?", "Adele attended the BRIT School for Performing Arts and Technology."),
    ("Which Adele song has a greeting as its title?", "Hello is an Adele song with a greeting as its title."),
    ("What is the title of Adele's album released after 25?", "The album Adele released after 25 is 30."),
    ("What is the title of Adele's album released before 25?", "The album Adele released before 25 is 21."),
    ("What is Adele's nationality?", "Adele is British."),
    ("What kind of performer is Adele?", "Adele is a singer-songwriter and recording artist."),
    ("Why are Adele's songs widely recognized?", "Adele's songs are widely recognized for emotional lyrics and strong vocal performances."),
    ("Which Adele song includes the title phrase Someone Like You?", "Someone Like You is one of Adele's well-known songs from the album 21."),
    ("Which Adele song includes the title phrase Set Fire to the Rain?", "Set Fire to the Rain is one of Adele's well-known songs from the album 21."),
    ("Why do Adele's albums often use numbers as titles?", "Adele's albums use numbers associated with periods of her life."),
    ("How is Adele usually described in relation to songwriting?", "Adele is usually described as a singer-songwriter who writes emotionally direct songs."),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Input KIF prompts.jsonl")
    ap.add_argument("--out", required=True, help="Output prompts_with_adele_retain.jsonl")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    base = Path(args.base)
    out = Path(args.out)
    if not base.exists():
        raise FileNotFoundError(base)
    if out.exists() and not args.overwrite:
        raise FileExistsError(f"{out} already exists. Pass --overwrite to replace it.")
    out.parent.mkdir(parents=True, exist_ok=True)

    text = base.read_text(encoding="utf-8")
    with out.open("w", encoding="utf-8") as f:
        f.write(text)
        if text and not text.endswith("\n"):
            f.write("\n")
        for prompt, response in ADELE_ROWS:
            f.write(json.dumps({"subject": "Adele", "prompt": prompt, "response": response}, ensure_ascii=False) + "\n")

    print(f"Wrote {out}")
    print(f"Copied base prompts from {base}")
    print(f"Appended Adele retain rows: {len(ADELE_ROWS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
