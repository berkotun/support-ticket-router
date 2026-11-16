"""Area B: regex normalization and code / entity extraction.

normalize(): lower-case, expand contractions ("can't" -> "can not"), replace
URLs / e-mails with placeholders, write money as words ("$50" -> "50 dollars"),
collapse repeated punctuation and letters ("helloooo!!!" -> "hello!").

extract_codes(): pulls structured codes out of the raw message with regular
expressions, so a support agent sees them at once:
    model codes   "DW-6534", "xr 200"      product / device model numbers
    flight codes  "UA 1234", "dl567"       airline code + flight number
    card endings  "card ending in 4321"    last 4 digits of a card
    amounts       "$250", "40 dollars"
    dates         "12/05", "march 3rd"
"""
import re

CONTRACTIONS = {
    "can't": "can not", "won't": "will not", "n't": " not", "'re": " are",
    "'s": " is", "'d": " would", "'ll": " will", "'ve": " have", "'m": " am",
}
RE_CONTRACTION = re.compile("|".join(re.escape(k) for k in sorted(CONTRACTIONS, key=len, reverse=True)))
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
RE_MONEY = re.compile(r"\$\s?(\d+(?:[.,]\d+)?)")
RE_REPEAT_PUNCT = re.compile(r"([!?.,])\1+")
RE_REPEAT_CHAR = re.compile(r"([a-z])\1{2,}")      # 3+ same letters -> 1 ("sooo" -> "so")
RE_KEEP = re.compile(r"[^a-z0-9\s'!?.,/-]")
RE_SPACES = re.compile(r"\s+")

AIRLINES = r"(?:aa|ua|dl|wn|b6|as|nk|f9|ba|lh|af|tk|ek|qr)"
CODE_PATTERNS = {
    "flight": re.compile(rf"\b{AIRLINES}\s?-?\s?(\d{{2,4}})\b", re.I),
    # upper-case letters + digits ("DW6534", "XR 200") or any letters with a hyphen ("dw-6534")
    "model_code": re.compile(r"\b(?:([A-Z]{1,4})[\s-]?(\d{3,6})|([A-Za-z]{1,4})-(\d{3,6}))\b"),
    "card_ending": re.compile(r"\bending (?:in|with) (\d{4})\b", re.I),
    "amount": re.compile(r"(\$\s?\d+(?:[.,]\d+)?|\b\d+(?:[.,]\d+)? (?:dollars|bucks|usd|euros?)\b)", re.I),
    "date": re.compile(r"\b(\d{1,2}/\d{1,2}(?:/\d{2,4})?|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.? \d{1,2}(?:st|nd|rd|th)?)\b", re.I),
}


def normalize(text: str) -> str:
    t = text.lower().replace("’", "'")
    t = RE_URL.sub(" url ", t)
    t = RE_EMAIL.sub(" email ", t)
    t = RE_MONEY.sub(r"\1 dollars", t)
    t = RE_CONTRACTION.sub(lambda m: CONTRACTIONS[m.group(0)], t)
    t = RE_REPEAT_PUNCT.sub(r"\1", t)
    t = RE_REPEAT_CHAR.sub(r"\1", t)
    t = RE_KEEP.sub(" ", t)
    return RE_SPACES.sub(" ", t).strip()


def extract_codes(text: str) -> dict:
    found = {}
    flights = {f"{m.group(0).upper().replace(' ', '').replace('-', '')}"
               for m in CODE_PATTERNS["flight"].finditer(text)}
    if flights:
        found["flight"] = sorted(flights)
    models = set()
    for m in CODE_PATTERNS["model_code"].finditer(text):
        letters, digits = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        code = f"{letters.upper()}-{digits}"
        if code.replace("-", "") not in flights:
            models.add(code)
    if models:
        found["model_code"] = sorted(models)
    for key in ("card_ending", "amount", "date"):
        hits = sorted({m.group(1) for m in CODE_PATTERNS[key].finditer(text)})
        if hits:
            found[key] = hits
    return found


def tokenize(text: str) -> list[str]:
    """Word tokens of the normalized text (punctuation removed)."""
    return re.findall(r"[a-z0-9']+", normalize(text))


if __name__ == "__main__":
    for s in ["My dishwasher DW-6534 won't drain!!!", "is UA 1234 on time?",
              "I can't use my card ending in 4321, it was declined for $250",
              "helloooo there", "book a table for march 3rd"]:
        print(f"{s!r}\n   -> {normalize(s)!r}  codes={extract_codes(s)}")
