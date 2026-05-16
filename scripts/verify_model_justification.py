import re
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "model_justification.tex"
BIB = ROOT / "refs.bib"
OUT = ROOT / "model_justification_verification.md"


def read_text(p: Path):
    return p.read_text(encoding="utf-8", errors="ignore")


def strip_latex(s: str):
    # remove common LaTeX commands and math
    s = re.sub(r"\\cite\{.*?\}", "[CITATION]", s)
    s = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", "", s)
    s = re.sub(r"\$.*?\$", "", s)
    s = re.sub(r"%.*", "", s)
    return s


def split_sentences(text: str):
    # naive sentence split on punctuation
    parts = re.split(r"(?<=[.!?])\s+", text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts


def parse_bib(bib_text: str):
    entries = {}
    current_key = None
    buffer = []
    for line in bib_text.splitlines():
        m = re.match(r"@\w+\{([^,]+),", line)
        if m:
            if current_key:
                entries[current_key] = "\n".join(buffer)
            current_key = m.group(1).strip()
            buffer = [line]
        else:
            if current_key:
                buffer.append(line)
    if current_key:
        entries[current_key] = "\n".join(buffer)
    return entries


def find_citations(tex: str):
    keys = re.findall(r"\\cite[t]?[a-zA-Z]*\{([^}]+)\}", tex)
    all_keys = []
    for k in keys:
        for part in k.split(","):
            all_keys.append(part.strip())
    return set(all_keys)


def sentence_verdict(sentence: str, bib_entries, root: Path):
    verdict = "Needs-evidence"
    evidence = []
    # citation present?
    cites = re.findall(r"\\cite[t]?[a-zA-Z]*\{([^}]+)\}", sentence)
    if cites:
        for c in cites:
            for key in [k.strip() for k in c.split(",")]:
                if key in bib_entries:
                    verdict = "Verified (external)"
                    evidence.append(f"refs.bib:{key}")
                else:
                    evidence.append(f"citation {key} not found in refs.bib")
        return verdict, evidence

    # local file mentions
    files = re.findall(r"[\w\-./]+\.(txt|jsonl|py|md|pth|pdf|tex)", sentence)
    for f in files:
        p = root / f
        if p.exists():
            verdict = "Verified (local)"
            evidence.append(str(p.relative_to(root)))

    # numeric checks (e.g., 2997)
    nums = re.findall(r"\b\d{3,6}\b", sentence)
    for n in nums:
        # check common dataset size
        if n == "2997":
            p = root / "dataset" / "im2gps3k_rgb_images" / "meta.jsonl"
            if p.exists():
                verdict = "Verified (local)"
                evidence.append(str(p.relative_to(root)))

    if evidence:
        return verdict, evidence

    # fallback: if sentence contains key method names, mark as Supported-by-citation if present in refs
    keywords = [
        "GroundingDINO",
        "CLIP",
        "FAISS",
        "LoRA",
        "Qwen2-VL",
        "LLaVA",
        "GeoGuessr",
    ]
    for k in keywords:
        if k in sentence:
            # check if any bib entry mentions k
            hit = any(k in v for v in bib_entries.values())
            if hit:
                return "Verified (external)", [f"refs.bib (mentions {k})"]

    return verdict, ["no local or bib evidence found"]


def main():
    tex = read_text(TEX)
    bib_text = read_text(BIB) if BIB.exists() else ""
    bib_entries = parse_bib(bib_text)

    clean = strip_latex(tex)
    sentences = split_sentences(clean)

    with OUT.open("w", encoding="utf-8") as fo:
        fo.write("# model_justification_verification\n\n")
        fo.write(
            "This file lists each extracted sentence from `model_justification.tex` with an automated preliminary verdict and evidence pointers.\n\n"
        )
        for i, s in enumerate(sentences, 1):
            verdict, evidence = sentence_verdict(s, bib_entries, ROOT)
            fo.write(f"## Sentence {i}\n")
            fo.write(f"> {s}\n\n")
            fo.write(f"- **Verdict:** {verdict}\n")
            for e in evidence:
                fo.write(f"- **Evidence:** {e}\n")
            fo.write("\n")

    print("Wrote", OUT)


if __name__ == "__main__":
    main()
