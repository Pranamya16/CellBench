"""Plain-language explanations from Qwen 2.5 (0.5B, served by Ollama), with a hard number check and a template fallback.

Qwen only ever sees a small FACTS record. Whatever it writes is rejected unless:
    - every number in it appears in the facts (so it cannot invent a figure),
    - it says the patient is synthetic,
    - it makes no diagnosis or disease-probability claim,
    - it is short.
If it fails twice, a fixed template sentence built from the same facts is shown instead, and the page says so.
"""
"""Plain-language explanations: a correct DRAFT built from the facts, optionally REWRITTEN by Qwen 2.5 (0.5B, via Ollama).

The draft is always accurate. Qwen only rewrites it in friendlier words, and its rewrite is accepted only if:
    - every number is in the facts AND both scores are quoted,
    - it stays close to the draft (it cannot add new content),
    - it does not contradict the facts (e.g. says the views "agreed" when they did not),
    - it says the patient is synthetic, makes no diagnosis or disease-probability claim, and is short.
Otherwise the draft itself is shown and the page says so. A 0.5B model passes a number check while writing nonsense, so the
meaning checks matter as much as the number check.
"""
import json
import re
import time

import httpx

OLLAMA_URL = "http://localhost:11434"
MODEL = "qwen2.5:0.5b"
BANNED = re.compile(r"diagnos|cancer|leukemi|leukaemi|you have|has mds|probability of|chance of (disease|cancer|illness)|treat|prescrib|\bcure\b", re.I)


def _pct(x):
    return f"{round(float(x) * 100)}"


def verdict(score, signal):
    """What a score amounts to, given the truth. Scores between 40 and 60 count as undecided."""
    if signal is None:   # uploaded file: the truth is not known, so say which way the score leans
        return "leans signal-like" if score >= 60 else "leans normal-like" if score <= 40 else "is undecided"
    if signal:
        return "found the change" if score >= 60 else "missed the change" if score <= 40 else "was undecided"
    return "correctly stayed low" if score <= 40 else "raised a false alarm" if score >= 60 else "was undecided"


def lean(score):
    """Which way a 0-100 detector score points."""
    return "planted" if score >= 60 else "normal" if score <= 40 else "undecided"


def make_facts(p, limits=None):
    """p: patient dict from Scorer.package. Every figure Qwen may quote is listed in `allowed`."""
    sc_cells, sc_rep = round(p["score_cells"] * 100), round(p["score_report"] * 100)
    if p.get("upload"):
        change = f"uploaded file {p['name']} with {p['lab']['events']} cells; the truth is not known"
    elif p["signal"]:
        if p["kind"] == "shift":
            change = f"every neutrophil was made {_pct(p['drop'])}% less granular"
        elif p["kind"] == "spread":
            change = f"the neutrophil cluster was widened (matched to {_pct(p['frac'])}% of neutrophils changed by {_pct(p['drop'])}%)"
        else:
            change = f"{_pct(p['frac'])}% of the neutrophils were made {_pct(p['drop'])}% less granular"
    else:
        change = "nothing was planted: this is a normal synthetic patient"
    facts = dict(
        patient="uploaded" if p.get("upload") else "synthetic", truth="unknown" if p.get("upload") else ("signal" if p["signal"] else "normal"), change=change,
        planted_cells=p["planted"], neutrophils=p["neutrophils"],
        score_full_cell_data=sc_cells, score_report_averages=sc_rep,
        cells_verdict=verdict(sc_cells, p["signal"]), report_verdict=verdict(sc_rep, p["signal"]))
    if limits:
        facts.update(limits)
    ev = p.get("evidence")
    if ev:
        facts.update(lean_full=lean(sc_cells), lean_report=lean(sc_rep), clues=ev,
                     warning=" ".join(p.get("warnings") or []))
    nums = set()
    for v in facts.values():
        nums.update(re.findall(r"\d+(?:\.\d+)?", str(v)))
    for c in (ev or {}).values():
        nums.update(re.findall(r"\d+(?:\.\d+)?", " ".join(str(c[k]) for k in ("value", "lo", "hi"))))
    nums.update({"0", "100", "50"})   # scales that appear in the wording: scores run 0 to 100, 50 is the midpoint
    facts["allowed_numbers"] = sorted(nums, key=float)
    return facts


STOP = set("the a an and or of to in on is are was were be been it its this that these those with for as at by from but not no than then so if "
           "can may will would should has have had do does did into over under out up about which who whom what when where while".split())


def template(f):
    """The accurate draft. Also the fallback."""
    c, r = f["cells_verdict"], f["report_verdict"]
    if f["truth"] == "unknown":
        return (f"This is an {f['change']}. The full cell data model scored it {f['score_full_cell_data']} out of 100 and {c}; "
                f"the report averages model scored it {f['score_report_averages']} out of 100 and {r}. "
                "These models were trained only on simulated patients, so their scores mean little for other kinds of data. "
                "These are detector scores, not a diagnosis.")
    first = f"This is a synthetic patient: {f['change']}."
    second = (f"The full cell data model scored it {f['score_full_cell_data']} out of 100 and {c}; "
              f"the report averages model scored it {f['score_report_averages']} out of 100 and {r}.")
    if f["truth"] == "normal":
        third = "Scores should stay low for a normal patient."
    elif c == r == "found the change":
        third = "The change was big enough for both views."
    elif c == "found the change":
        third = "Averages can hide a small group of changed cells, which the cell map still shows."
    elif r == "found the change":
        third = "Here the report averages did better than the full cell data."
    elif c == r == "missed the change":
        third = "This change is too small for either view to see."
    else:
        third = "Neither view found the change clearly."
    return f"{first} {second} {third} These are detector scores on made-up data, not a diagnosis."


def words(s):
    return [w for w in re.findall(r"[a-z]{4,}", s.lower()) if w not in STOP]


def check(text, f, draft=None):
    """Returns (ok, reason)."""
    t = text.strip()
    draft = draft or template(f)
    if not t:
        return False, "empty"
    if len(t.split()) > 110 or len(re.findall(r"[.!?](?:\s|$)", t)) > 6:
        return False, "too long"
    low = t.lower()
    if f["truth"] == "unknown":
        if "simulated" not in low:
            return False, "does not say the models only know simulated patients"
    elif "synthetic" not in low and "made-up" not in low:
        return False, "does not say synthetic"
    if BANNED.search(re.sub(r"not a diagnosis|no diagnosis|not diagnos\w+", "", t, flags=re.I)):
        return False, "banned wording"
    strip = lambda s: re.sub(r"out of 100", "", s)
    nums = re.findall(r"\d+(?:\.\d+)?", strip(t))
    if nums != re.findall(r"\d+(?:\.\d+)?", strip(draft)):
        return False, "numbers differ from the draft (set or order)"
    allowed = set(f["allowed_numbers"])
    for n in nums:
        if n not in allowed:
            return False, f"number {n} not in the facts"
    for need in (str(f["score_full_cell_data"]), str(f["score_report_averages"])):
        if need not in nums:
            return False, f"score {need} not quoted"
    c, r = f["cells_verdict"], f["report_verdict"]
    if c != r and re.search(r"\b(both|agreed|agree|same call|same result|either)\b", low):
        return False, "says the views were alike when they were not"
    if c == r and re.search(r"\b(only|disagree|whereas)\b", low):
        return False, "says the views differed when they did not"
    KW = {"leans signal-like": r"lean|signal-like|toward|tends|points",
          "leans normal-like": r"lean|normal-like|toward|tends|points",
          "found the change": r"found|picked up|detect|caught|spotted|noticed|saw|see",
          "missed the change": r"miss|did not|didn't|failed|overlooked|too small|not see|neither",
          "was undecided": r"undecided|unsure|not sure|unclear|middle|uncertain|neither",
          "is undecided": r"undecided|unsure|not sure|unclear|middle|uncertain|neither",
          "correctly stayed low": r"low|normal|correct|stayed|right",
          "raised a false alarm": r"false alarm|wrongly|mistaken|raised"}
    for v in {c, r}:
        if not re.search(KW[v], low):
            return False, f"does not convey that a view {v}"
    if f["truth"] == "unknown" and re.search(r"found the change|missed the change|correct|false alarm", low):
        return False, "claims a result the truth does not support"
    if f["truth"] == "normal" and "normal" not in low:
        return False, "does not say the patient is normal"
    if f["truth"] == "signal" and not re.search(r"changed|less granular|planted|widened|signal", low):
        return False, "does not describe the planted change"
    if re.search(r"average (cell )?count|accuracy and simplicity|balance between|perfect|significant|\bdrugs?\b|tiny|usual|medic|therap|infect|\bsick|healthy|\bill\b|blood cell|white blood", low):
        return False, "invented statement"
    ow, dw = set(words(t)), set(words(draft))
    FRIENDLY = {"also", "means", "meaning", "like", "very", "much", "just", "every", "each", "showed", "shows", "show", "shown", "found", "tell", "says", "said",
                "made", "make", "made-up", "real", "someone", "overall", "result", "results", "patients", "patient", "score", "scores", "scored", "sure", "certain"}
    novel = [w for w in ow if w not in dw and w not in FRIENDLY]
    if len(novel) > 2:
        return False, "adds new words: " + ", ".join(sorted(novel)[:4])
    if not ow or len(ow & dw) / len(ow) < 0.55:
        return False, "adds content that is not in the facts"
    if len(ow & dw) / max(1, len(dw)) < 0.35:
        return False, "drops too much of the draft"
    return True, "ok"


SYSTEM = (
    "You are a careful writing assistant for a research demo about blood-cell data. You rewrite a short technical text in plain, "
    "friendly English for someone with no medical or statistics background. Rules: keep every number exactly as written and in the same "
    "order; keep the meaning exactly; do not add any new facts, numbers, causes, advice or examples; never mention disease, diagnosis, "
    "drugs or treatment; keep the closing sentence about detector scores and diagnosis; reply with the rewritten text only, "
    "at most 5 short sentences.")


def prompt(f):
    """The user message: just the accurate draft to rewrite."""
    return "Rewrite this text:\n\n" + template(f)


def explain(f, attempts=3, timeout=40):
    """Returns dict(text, source, checked, reason, ms). Never raises."""
    if f.get("clues"):
        return explain_paragraph(f)
    t0 = time.time()
    reason = "Qwen not reached"
    for k in range(attempts):
        try:
            r = httpx.post(f"{OLLAMA_URL}/api/chat", timeout=timeout, json=dict(
                model=MODEL, stream=False, options=dict(temperature=(0.3, 0.1, 0.5)[k % 3], num_predict=120, top_p=0.9, seed=11 + k),
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(f)}]))
            text = r.json()["message"]["content"].strip()
        except Exception as e:  # server down, model missing, timeout
            reason = f"Qwen not reached ({type(e).__name__})"
            break
        ok, reason = check(text, f)
        if ok:
            return dict(text=text, source="Qwen 2.5 0.5B", checked=True, reason=reason, ms=int((time.time() - t0) * 1000))
    return dict(text=template(f), source="template", checked=True, reason=reason, ms=int((time.time() - t0) * 1000))


# ---------------------------------------------------------------------------------------------------------------------------------------
# One-paragraph "why" explanation (used when the facts carry clues): a correct draft from the facts, rewritten by the chat-size Qwen model.
# The networks cannot give an exact reason for a score, so the paragraph states the leaning and both scores, what the models look for, and the readable
# clues in this file's neutrophils placed against normal synthetic patients. Where the clues show nothing clear, it says so.
# ---------------------------------------------------------------------------------------------------------------------------------------
PARA_MODEL = "qwen2.5:3b"
_PICK = {"t": 0.0, "m": MODEL}
LEAN_TEXT = {"planted": "looks like a patient with a planted neutrophil change", "normal": "looks like a normal patient", "undecided": "is hard to call"}
REPORT_LEAN = {"planted": "leans toward a planted change", "normal": "leans toward normal", "undecided": "is hard to call"}
POS = {"below": "below the normal range", "above": "above the normal range", "within": "inside the normal range"}


def para_model():
    if time.time() - _PICK["t"] > 60:
        _PICK["t"] = time.time()
        try:
            names = {m["name"] for m in httpx.get(f"{OLLAMA_URL}/api/tags", timeout=3).json().get("models", [])}
        except Exception:
            names = set()
        _PICK["m"] = PARA_MODEL if PARA_MODEL in names else MODEL
    return _PICK["m"]


CLUE = {"ssc": "the neutrophil side-scatter average", "tail": "the share of neutrophils far on the low-scatter side of their own cluster", "sd": "the width of the neutrophil cluster"}


def _fmt(k, c):
    u = "%" if k == "tail" else ""
    return f"{CLUE[k]} is {c['value']}{u}, {POS[c['pos']]} ({c['lo']}{u} to {c['hi']}{u})"


def hits(f):
    return [k for k in ("ssc", "tail", "sd") if f["clues"][k]["hit"]]


def clue_lines(f):
    h = hits(f)
    if not h:
        return ["no simple clue (side-scatter average, low-scatter share, cluster width) points to less granular neutrophils; all are inside the normal range or point the other way"]
    return [_fmt(k, f["clues"][k]) + ", which points to less granular neutrophils" for k in h]


def para_template(f):
    """The accurate one-paragraph draft. Also the fallback."""
    up = f["truth"] == "unknown"
    full, rep = f["score_full_cell_data"], f["score_report_averages"]
    lf, lr = f["lean_full"], f["lean_report"]
    s1 = f"The full cell data model scored this {'file' if up else 'synthetic patient'} {full} out of 100, so it {LEAN_TEXT[lf]}."
    if lf == lr:
        s1 += f" The report averages model scored it {rep} and points the same way."
    else:
        s1 += f" The report averages model scored it {rep} and {REPORT_LEAN[lr]}, because averages can hide a small group of changed cells."
    s2 = "The models look for a group of neutrophils with lower side scatter, meaning less granular."
    h = hits(f)
    if h:
        s3 = "Clues in the cells: " + "; ".join(_fmt(k, f["clues"][k]) for k in h) + "."
        if lf == "normal":
            s3 += " That is only a mild clue, and the full cell data model did not find enough of a pattern to call it a planted change."
    else:
        s3 = ("No simple clue (side-scatter average, low-scatter share, cluster width) points to less granular neutrophils, so any call here rests on finer patterns in the cell maps." if lf == "planted"
              else "No simple clue (side-scatter average, low-scatter share, cluster width) points to less granular neutrophils.")
    s4 = ("These models only know simulated patients, so this is a detector score, not a diagnosis." if up else "This is made-up data and a detector score, not a diagnosis.")
    s5 = (" " + f["warning"]) if f.get("warning") else ""
    return f"{s1} {s2} {s3} {s4}{s5}"


PARA_SYSTEM = (
    "You explain a blood-cell detector's result to someone with no medical or statistics background. You get FACTS and a DRAFT paragraph. Rewrite the DRAFT as ONE friendly, "
    "plain-English paragraph of at most 95 words. Rules: keep every number exactly as written; keep which way each score points and whether each clue is inside, below or above "
    "the normal range; keep the point that this is a detector score on made-up (or simulated) data and not a diagnosis; add no new facts, numbers, causes, advice or examples; "
    "never say the patient has or does not have a disease; do not use bullet points or line breaks; never call anything an issue or a problem. "
    "You MUST copy these exact phrases from the DRAFT unchanged: the phrase after 'so it' in the first sentence, 'points the same way' or the explanation of why the report averages differ, "
    "and every clue sentence including its numbers and 'inside/below/above the normal range'. Only simplify the other wording. Reply with the paragraph only.")


def para_check(text, f):
    t = text.strip()
    if not t or "\n" in t.replace("\r", ""):
        return False, "empty or more than one paragraph"
    if len(t.split()) > 125:
        return False, "too long"
    low = t.lower()
    if "synthetic" not in low and "made-up" not in low and "simulated" not in low:
        return False, "does not say the data is made up or simulated"
    if BANNED.search(re.sub(r"not a diagnosis|no diagnosis|not diagnos\w+|not (a )?medical", "", t, flags=re.I)):
        return False, "banned wording"
    allowed = set(f["allowed_numbers"])
    nums = re.findall(r"\d+(?:\.\d+)?", t)
    for n in nums:
        if n not in allowed:
            return False, f"number {n} not in the facts"
    for need in (str(f["score_full_cell_data"]), str(f["score_report_averages"])):
        if need not in nums:
            return False, f"score {need} not quoted"
    lf, lr = f["lean_full"], f["lean_report"]
    if not re.search({"planted": r"planted (neutrophil )?change", "normal": r"normal patient", "undecided": r"hard to call"}[lf], low):
        return False, "does not keep the verdict phrase"
    for k in hits(f):
        if f"is {f['clues'][k]['value']}" not in low and str(f["clues"][k]["value"]) not in t:
            return False, "drops a clue"
        if POS[f["clues"][k]["pos"]] not in low:
            return False, "drops where a clue sits against the normal range"
    if lf == "planted" and (not re.search(r"planted (neutrophil )?change", low) or re.search(r"looks like a normal|normal patient|looks normal|no issues?|nothing (wrong|unusual)|no problem|healthy|\bfine\b|all clear", low)):
        return False, "does not convey that the full cell model leans planted"
    if lf == "normal" and (not re.search(r"normal patient", low) or re.search(r"looks like a patient with a planted|has a planted|planted (neutrophil )?change (is|was) (found|seen|present)|\bproblem\b|\bissues?\b(?! with)", low)):
        return False, "does not convey that the full cell model leans normal"
    if lf == lr and re.search(r"\bhowever\b|\bwhereas\b|disagree|\bdiffer|on the other hand", low):
        return False, "says the two models differ when they agree"
    if lf != lr and re.search(r"same way|\bagree|\bboth\b|also points|pointing to the same|same (call|problem|result)", low):
        return False, "says the two models agree when they differ"
    if lf == "undecided" and not re.search(r"hard to call|undecided|unsure|unclear|not sure|borderline|in between|middle", low):
        return False, "does not convey that the call is unclear"
    if not hits(f) and re.search(r"below the normal|above the normal|outside the normal|lower than normal|higher than normal|wider than|unusually", low):
        return False, "claims a clue when none points to less granular neutrophils"
    if re.search(r"no (simple )?clues?( \w+)? (is|are) (inside|within)", low):
        return False, "garbled clue statement"
    if hits(f) and lf != "normal" and re.search(r"no (simple )?clues?|none of the (simple )?clues|nothing (unusual|points)", low):
        return False, "says there is no clue when there is one"
    for k in ("ssc", "tail", "sd"):
        if not f["clues"][k]["hit"] and k in hits(f):
            pass
    if re.search(r"\bdrugs?\b|medic|therap|infect|\bsick|\bill\b|cancer|perfect|significant", low):
        return False, "invented statement"
    return True, "ok"


def explain_paragraph(f, attempts=2, timeout=60):
    t0 = time.time()
    draft = para_template(f)
    reason, model = "model not reached", para_model()
    body = "FACTS:\n" + "\n".join("- " + x for x in clue_lines(f)) + f"\n- full cell data model score: {f['score_full_cell_data']} out of 100\n- report averages model score: {f['score_report_averages']} out of 100\n\nDRAFT:\n{draft}"
    for k in range(attempts):
        try:
            r = httpx.post(f"{OLLAMA_URL}/api/chat", timeout=timeout, json=dict(
                model=model, stream=False, keep_alive="30m", options=dict(temperature=(0.3, 0.0)[k], num_predict=220, seed=3 + k),
                messages=[{"role": "system", "content": PARA_SYSTEM}, {"role": "user", "content": body}]))
            text = " ".join(r.json()["message"]["content"].split())
        except Exception as e:
            reason = f"model not reached ({type(e).__name__})"
            break
        ok, reason = para_check(text, f)
        if ok:
            return dict(text=text, source=model, checked=True, reason=reason, ms=int((time.time() - t0) * 1000))
    return dict(text=draft, source="template", checked=True, reason=reason, ms=int((time.time() - t0) * 1000))
