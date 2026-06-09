import re

def wrap_arpabet(match):
    content = match.group(1).strip()
    if re.fullmatch(r"[A-Z0-9\s]+", content):
        return f"[[{content}]]"
    return match.group(0)

text = "[thought-hmm] maintain your [R IH0 L EY1 SH AH0 N Z] with [breath] [V AH1 L N ER0 AH0 B AH0 L]."
processed = re.sub(r"\[([^\]]+)\]", wrap_arpabet, text)
print(f"Result: {processed}")
