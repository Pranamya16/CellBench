"""Grounded chatbot: answers only from the project notes (knowledge.py), shows its sources, and refuses medical questions.

Pipeline: refuse unsafe or off-topic questions -> retrieve the 3 most relevant notes (TF-IDF) -> Qwen writes 2 to 3 sentences from only
those notes -> checks (numbers, banned wording, no new content) -> otherwise show the best note itself. An answer is never ungrounded.
"""
import re
import time

import httpx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import explain
import knowledge

MIN_SCORE = 0.10
EMB_W, EMB_FLOOR, EMB_SCOPE = 1.5, 0.45, 0.58
# Answers are the project note itself, word for word. A 0.5B model passed every check while garbling two facts into a false one
# ("built by the hackathon"), so it is switched off here; Qwen still explains scored patients on the main page, from a checked draft.
USE_QWEN = True
CHAT_MODEL = "qwen2.5:3b"          # answers; falls back to the small explain model if it is not installed
EMBED_MODEL = "nomic-embed-text"   # finds the right notes by meaning; falls back to word matching if it is not installed
YESNO = re.compile(r"^\s*(is|are|was|were|do|does|did|can|could|will|would|should|has|have|had|am)\b", re.I)
NEG = re.compile(r"\b(no|not|never|cannot|nor|neither|none|without|nothing)\b|n't", re.I)
SENT = re.compile(r"(?<=[.!?])\s+")
REFUSE = re.compile(r"\bdo i have\b|\bam i\b|\bmy (own )?(blood|result|results|test|sample|doctor|health|child|mother|father|body)\b|\bsafe to use\b|\bis it safe\b|\bshould i\b|"
                    r"\b(treatment|therapy|medication|prescribe|prognosis|symptoms?|cure|dosage)\b|\bhow (do|can) i (treat|cure)\b", re.I)
INJECT = re.compile(r"ignore (all|any|the|previous|your)|system prompt|your instructions|reveal your|jailbreak|pretend (to be|you)|act as", re.I)
REFUSAL_TEXT = ("I can't help with medical questions, diagnosis or treatment. CellBench is a research demo on made-up data and gives no medical advice. "
                "I can explain what the scores mean, how the data was made, or what the results show.")
OUT_TEXT = ("That is outside what I can help with. I only answer questions about the CellBench project: its data, models, results, limits and how to run it, "
            "using its own notes. Try one of the example questions.")
HELLO = re.compile(r"^\s*(hi|hello|hey|thanks|thank you|ok|okay|good (morning|afternoon|evening))\W*$", re.I)
OFFTASK = re.compile(r"\b(write|compose|draft|generate|create|give me|tell me)\b[^.?]{0,40}\b(poem|story|essay|song|joke|email|letter|code|script|program|recipe|cv|resume|haiku|lyrics)\b|"
                     r"\b(joke|poem|recipe|weather|forecast|horoscope|lottery|stock price|bitcoin|football|cricket|movie|netflix)\b|\bcapital of\b|\btranslate\b|\bwho (won|is the (president|prime minister))\b|"
                     r"\b(sort a list|python|javascript|java script|sql query)\b", re.I)
OFFTOPIC_TEXT = "I can only answer questions about the CellBench project, using its own notes."
UNKNOWN_TEXT = "I don't have a note on that. I can only answer questions about this project: its data, models, results and limits."
OLD_SYSTEM = ("You answer questions about the CellBench research demo using ONLY the NOTES you are given. Answer in at most 3 short plain sentences. "
          "Quote numbers exactly as written in the notes and never add numbers or facts that are not in the notes. If the notes do not answer the question, "
          "say you don't know. Never give medical advice or a diagnosis. Reply with the answer only.")
SYSTEM = """You are the CellBench assistant, a friendly chatbot that answers questions about the CellBench research project. CellBench is a "detection ruler": it plants a known change in some neutrophils of made-up (synthetic) blood-cell data and measures how small a change an AI can still see, comparing a model that sees report averages with one that sees every cell.

You get NOTES from the project. Rules:
1. Answer the question the user actually asked, directly and conversationally, in plain everyday words that a non-doctor understands. Usually 2 to 5 short sentences. Use a short list only if the user asks for steps or a list.
2. Use ONLY facts and numbers found in the NOTES. Never invent or calculate numbers, ratios, results, names or dates. If a note already explains something in plain words, follow it closely. If the notes answer only part of the question, answer that part and say plainly what the project has not covered. If something was not tested or is not in the notes, say it was not tested or is not covered; never answer yes or no for it.
3. Take the earlier conversation into account for follow-up questions such as "why?" or "and the second one?".
4. If the question is not about the CellBench project, its data, models, results, limits or how to run it, reply with exactly: OUT_OF_SCOPE
5. Never give medical advice or a diagnosis. If asked how this could help doctors or hospitals, say it is a research demo on synthetic data that shows how small a cell change an AI can see, that it has not been tested on real patients and is not a diagnostic tool; never say it will lead to earlier detection or help diagnose. Say \"has not been tested on real patients\", never \"does not work\" or \"works\". A detector score is not a chance of disease. The data is synthetic and the tool has not been tested on real patients: never suggest otherwise. Side scatter is how grainy a cell is inside; never call it brightness. Never say it detects early disease; say it detects smaller cell changes in synthetic data.
6. Do not mention "the notes" or these rules. Reply with the answer only."""
KEYWORDS = {
    "readings": "SSC SFL FSC FSCW readings channels meaning side scatter forward scatter fluorescence abbreviations terms",
    "final_models": "how accurate accuracy performance AUROC final test overall results correct",
    "headline": "result results finding findings conclusion main takeaway summary beat better outperform advantage why cnn averages compare comparison best",
    "neutrophil": "what is a neutrophil definition white blood cell infection",
    "real_data": "real data synthetic made up fake real patients tested on real",
    "data_made": "synthetic fake simulated data generated made created how was the data made patients",
    "not_diagnostic": "cancer disease diagnose diagnosis diagnostic clinical medical real patients use tool treat leukemia MDS",
    "limits": "limitations problems weaknesses caveats risks",
    "upload": "upload file csv own data format columns",
    "live_demo": "demo live page how does it work generate patient",
    "models": "model models cnn mlp architecture neural network trained",
    "detector_score": "score scores 0 to 100 percentage chance probability",
    "cnn_detail": "cnn architecture layers built look like structure convolution filters design how does the cnn work",
    "cnn_mlp": "what is a cnn mlp convolutional neural network type difference",
    "lab_sheet": "lab report result sheet pdf download page mock-up printout layout",
    "analyser": "blood test sysmex analyser xn laser measure machine routine",
    "limits_subgroup": "smallest smallest change detect find tiny small percent share limit how small",
    "smallest_tested": "can it find 1% 0.5% half percent smaller tiniest smallest tested lowest minimum",
    "hackathon": "who built made created author team when where date hackathon leeds society event competition",
    "not_validated": "validated validation validate clinically proven peer reviewed approved certified regulated ready clinical use",
    "speed": "fast slow long time quick seconds latency",
    "ai_tools": "ai tools claude qwen ollama kaggle used disclosure",
}
ABBREV = {"accurate": "accuracy final models AUROC", "accuracy": "accuracy final models AUROC", "best": "main result headline models", "better": "main result headline", "beat": "main result headline", "good": "accuracy final models",
          "ssc": "SSC side scatter", "sfl": "SFL side fluorescence", "fsc": "FSC forward scatter", "fscw": "FSCW forward scatter width", "cnn": "CNN model", "mlp": "MLP model"}
SIMPLE = re.compile(r"simpl|plain|layman|layperson|eli5|easy|easier|not a (doctor|scientist|expert)|non-?(medical|technical)|no medical|jargon|in short|summar", re.I)
STYLE = re.compile(r"\b(in )?(simpler|simple|plain|easy|easier|everyday)( english| language| words| terms)?\b|\blike (i'?m|i am|you'?re|you are) (not )?(a |an )?(doctor|scientist|expert|child|kid|5)\w*|\bso (i|we) (should )?understand\b[^.?]*|"
                   r"\b(i'?m|i am) not (a |an )?(doctor|scientist|expert)\b|\bnot (a |an )?(doctor|scientist|expert)\b", re.I)
CLINICAL_Q = re.compile(r"\b(help|helps|use|used|useful|usable|replace|benefit|for)\b[^?.]{0,25}\b(doctors?|clinicians?|hospitals?|nhs|clinics?)\b|\bvalidat\w*|\bclinical\b|real[- ]world|\bready for\b|\bpeer.review|\bethic", re.I)
GROUNDED_MIN = 0.5
WIDE = {"blood", "type", "cell", "white", "patient", "data", "test", "model", "work", "result", "project", "tool", "main", "different", "differ"}   # words so common here that sharing one proves nothing
CAUTION = re.compile(r"\breal\b|\bpatients?\b|diagnos|clinic|hospital|doctor|\bsafe\b|work(s)? (on|in|for)|\bdisease|\bcancer|\bnhs\b|\bproven\b|\bvalidated?\b", re.I)
# claims the project must never make, even softly ("could one day lead to earlier detection", "helps doctors diagnose")
OVERCLAIM = re.compile(r"\bearl(y|ier)\b[^.]{0,30}\b(detect\w*|diagnos\w*|disease|warning)|\b(detect|diagnos|identif|predict)\w*[^.]{0,30}\b(disease|cancer|leukaemia|leukemia|mds)\b(?![^.]{0,40}\b(not|no|never|cannot)\b)|"
                       r"\bone day\b|\b(help|helps|helping)\b[^.]{0,20}\b(doctors?|clinicians?)\b[^.]{0,20}\b(diagnos|detect|identif)", re.I)
COMMON = {w for w in "also about because better then there their they them what when which while would could should still only other between without within across among".split()}
GENERIC = set("tell give explain describe show know mean means meant work works tool use used using make makes made write code list help please thing things simple simpler simply plain "
              "words word language easy easier like want need would could able yes thanks hello more less some many much any all tool thing good bad new old big small first last best better beat machine laptop".split())
STOP = set("the a an and or of to in on is are was were be been it its this that these those with for as at by from but not no than then so if can may will would should has have "
           "had do does did into over under out up about which who whom what when where while how why you your".split())
FRIENDLY = {"also", "means", "meaning", "like", "very", "much", "just", "every", "each", "shows", "show", "shown", "found", "says", "made", "make", "real", "result", "results",
            "score", "scores", "scored", "about", "answer", "notes", "note", "project", "demo", "because", "mostly", "usually", "simply"}


class Chat:
    def __init__(self):
        self.cards = knowledge.build_cards()
        self._qcache = {}
        self._emb, self._emb_key, self._model, self._model_t = None, None, explain.MODEL, 0.0
        self._index(self.cards)

    def _index(self, cards):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
        self.mat = self.vec.fit_transform([c["title"] + " " + c["title"] + " " + KEYWORDS.get(c["id"], "") + " " + c["text"] for c in cards])
        self.cur = cards

    @staticmethod
    def stem(w):
        w = w.lower()
        for suf in ("ing", "ed", "es", "ly", "s"):
            if len(w) > len(suf) + 3 and w.endswith(suf):
                return w[:-len(suf)]
        return w

    def topic_stems(self, text):
        return {self.stem(w) for w in re.findall(r"[a-z0-9]{3,}", text.lower()) if w not in STOP and w not in GENERIC}

    def in_scope(self, q, extra=None):
        """A question is about the project only if it uses at least one term that the notes (or their keywords) use."""
        if not hasattr(self, "vocab"):
            self.vocab = self.topic_stems(" ".join(c["title"] + " " + c["text"] + " " + KEYWORDS.get(c["id"], "") for c in self.cards))
        if (self.topic_stems(q) - WIDE) & self.vocab or CLINICAL_Q.search(q):
            return True
        base, qv = self._note_vectors(self.cards), self._embed([q])      # no shared word: still in scope if close in meaning to a note
        return bool(base is not None and qv is not None and float((base @ qv[0]).max()) >= EMB_SCOPE)

    def retrieve(self, q, extra=None, k=3):
        cards = self.cards + ([extra] if extra else [])
        if extra:
            self._index(cards)
        elif self.cur is not self.cards:
            self._index(self.cards)
        simple = bool(SIMPLE.search(q))
        q = STYLE.sub(" ", q) if simple else q          # how the user wants it explained is not what they are asking about
        qx = " ".join(ABBREV.get(w.lower(), w) for w in re.findall(r"[A-Za-z0-9.%]+", q))
        sc = linear_kernel(self.vec.transform([qx]), self.mat).ravel()
        sc = sc + self.meaning_scores(q if not simple else q, cards, extra)
        if re.search(r"cancer|disease|leuk|\bmds\b|clinical|diagnos|clinician|replace|hospital|\bnhs\b", q, re.I):
            sc[[c["id"] for c in cards].index("not_diagnostic")] += 0.4      # these questions are about what we do NOT claim
        if simple and re.search(r"result|finding|found|show|overall|project|work|summar|about|everything|tell me", q, re.I) and not re.search(r"auroc|score|gate|upload|noise|cnn|mlp|accuracy", q, re.I):
            sc[[c["id"] for c in cards].index("plain_summary")] += 0.6
        if re.search(r"(?<![\d.])(0?\.\d+|1) ?%|smallest|tiniest|lowest|minimum|how small|even smaller|smaller than", q, re.I):
            sc[[c["id"] for c in cards].index("smallest_tested")] += 0.5      # sizes below what we tested
        if re.search(r"\b(help|helps|use|used|useful|usable|replace|benefit|for)\b[^?.]{0,25}\b(doctors?|clinicians?|hospitals?|nhs|clinics?|patients?)\b", q, re.I):
            for cid, w in (("not_diagnostic", 0.5), ("future", 0.3)):
                sc[[c["id"] for c in cards].index(cid)] += w
        if re.search(r"real patients?|real data|\breal\b", q, re.I):
            sc[[c["id"] for c in cards].index("real_data")] += 0.5
        m = re.search(r"what (?:is|are|does) (?:a |an |the )?([a-z0-9-]{3,})", q, re.I)
        if m:                                                               # definition questions: favour the note named after the term
            term = m.group(1).lower().rstrip("s")
            for i, c in enumerate(cards):
                if term in c["title"].lower() and c["title"].lower().startswith(("what", "is ")):
                    sc[i] += 0.5
        if extra and re.search(r"\b(this|my|the) patient\b|\bjust scored\b|\bthis (score|result|file)\b|\bscored\b", q, re.I):
            sc[-1] += 0.5      # the user is asking about the patient they just scored
        order = sc.argsort()[::-1][:k]
        return [(cards[i], float(sc[i])) for i in order if sc[i] >= MIN_SCORE]

    @staticmethod
    def nums(s):
        return re.findall(r"\d+(?:\.\d+)?", s)

    @staticmethod
    def words(s):
        return {w for w in re.findall(r"[a-z]{4,}", s.lower()) if w not in STOP}

    def check(self, ans, question, notes):
        a = ans.strip()
        if not a or len(a.split()) > 85:
            return False, "empty or too long"
        if explain.BANNED.search(re.sub(r"\b(not|no|never|isn't|cannot|can't)\b[^.]{0,30}(diagnos\w+|medical advice|advice)|can't help with[^.]*", "", a, flags=re.I)):
            return False, "banned wording"
        allowed = set(self.nums(question))
        for n in notes:
            allowed |= set(self.nums(n["text"] + " " + n["title"]))
        for n in self.nums(a):
            if n not in allowed:
                return False, f"number {n} is not in the notes"
        if re.match(r"\s*(yes|no)\b", a, re.I):
            return False, "starts with yes or no"
        # every sentence must come almost entirely from ONE note sentence, so two facts cannot be blended into a false third
        pool = [(self.words(s), bool(NEG.search(s))) for n in notes for s in SENT.split(n["text"])]
        for sent in SENT.split(a):
            w = self.words(sent) - self.words(question) - FRIENDLY
            if len(w) < 2:
                continue
            ratio, neg = max(((len(w & p) / len(w), pn) for p, pn in pool), key=lambda z: z[0], default=(0, False))
            if ratio < 0.75:
                return False, f"sentence not grounded in one note sentence ({ratio:.2f}): {sent[:50]}"
            if bool(NEG.search(sent)) != neg:
                return False, f"says the opposite of its source sentence: {sent[:50]}"
        return True, "ok"

    def used(self, ans, notes):
        """The notes whose sentences the answer was built from (for the 'based on' line)."""
        out = []
        for n in notes:
            pool = [self.words(s) for s in SENT.split(n["text"])]
            for sent in SENT.split(ans):
                w = self.words(sent) - FRIENDLY
                if len(w) >= 2 and max((len(w & p) / len(w) for p in pool), default=0) >= 0.75:
                    out.append(n); break
        return out or notes[:1]

    # ---- embeddings (meaning-based retrieval) ----------------------------------------------------------------------------
    def _embed(self, texts):
        key = tuple(texts)
        if len(texts) == 1 and key in self._qcache:
            return self._qcache[key]
        out = self._embed_raw(texts)
        if len(texts) == 1 and out is not None:
            self._qcache = {**dict(list(self._qcache.items())[-30:]), key: out}
        return out

    def _embed_raw(self, texts):
        try:
            r = httpx.post(f"{explain.OLLAMA_URL}/api/embed", json=dict(model=EMBED_MODEL, input=texts), timeout=60)
            E = np.array(r.json()["embeddings"], dtype=np.float64)
            return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        except Exception:
            return None

    def _note_vectors(self, cards):
        key = tuple(c["id"] + c["text"] for c in cards)
        if self._emb_key != key:
            self._emb_key = key
            self._emb = self._embed([c["title"] + ". " + c["text"] for c in cards])
        return self._emb

    def meaning_scores(self, q, cards, extra):
        """How close each note is to the question in meaning (embedding cosine), scaled to add to the word-match score. Zeros if no embedder."""
        base = self._note_vectors(self.cards)
        if base is None:
            return np.zeros(len(cards))
        qv = self._embed([q])
        if qv is None:
            return np.zeros(len(cards))
        cos = list(base @ qv[0])
        if extra:
            ev = self._embed([extra["title"] + ". " + extra["text"]])
            cos.append(float(ev[0] @ qv[0]) if ev is not None else 0.0)
        return EMB_W * np.clip(np.array(cos) - EMB_FLOOR, 0, None)

    def warm(self):
        """Load the chat and embedding models into memory so the first question is not slow."""
        try:
            self._note_vectors(self.cards)
            httpx.post(f"{explain.OLLAMA_URL}/api/chat", timeout=120, json=dict(model=self.chat_model(), stream=False, keep_alive="30m", options=dict(num_predict=1), messages=[{"role": "user", "content": "hi"}]))
        except Exception:
            pass

    def chat_model(self):
        """The 3B model if installed, else the small one."""
        if time.time() - self._model_t > 60:
            self._model_t = time.time()
            try:
                names = {m["name"] for m in httpx.get(f"{explain.OLLAMA_URL}/api/tags", timeout=3).json().get("models", [])}
            except Exception:
                names = set()
            self._model = CHAT_MODEL if CHAT_MODEL in names else explain.MODEL
        return self._model

    def answer(self, question, facts=None, use_llm=USE_QWEN, history=()):
        t0 = time.time()
        q = " ".join(question.split())[:300]
        done = lambda text, source, notes=(), reason="": dict(answer=text, source=source, notes=[n["title"] for n in notes], note_text=(notes[0]["text"] if notes else ""),
                                                             reason=reason, ms=int((time.time() - t0) * 1000))
        hist = [h for h in list(history)[-6:] if h.get("role") in ("user", "assistant") and str(h.get("content", "")).strip()]
        if not q:
            return done("Ask me something about the project, for example: what does AUROC mean?", "unknown")
        if INJECT.search(q):
            return done(OFFTOPIC_TEXT, "unknown")
        if REFUSE.search(q):
            return done(REFUSAL_TEXT, "refusal")
        if HELLO.match(q):
            return done("Hi. Ask me about the project: its data, models, results, limits or how to run it.", "unknown")
        followup = bool(hist) and len(q.split()) <= 7 and not OFFTASK.search(q)   # "why?", "and the second one?"
        lastq = next((h["content"] for h in reversed(hist) if h["role"] == "user"), "")
        rq = (lastq + " " + q) if followup else q
        if OFFTASK.search(q) or not (followup or self.in_scope(q)):
            return done(OUT_TEXT, "outofscope")
        extra = knowledge.patient_card(facts)
        hits = self.retrieve(rq, extra, k=4)
        if not hits:
            return done(UNKNOWN_TEXT, "unknown")
        notes = [h[0] for h in hits]
        if notes[0]["id"] == "plain_summary":   # a plain-words summary was asked for: the model gets only that note and follows it closely
            notes = notes[:1]
        best = notes[0]["text"]
        if not use_llm:
            return done(best, "note", notes[:1], "no language model")
        model = self.chat_model()
        reason = "model not reached"
        body = "NOTES:\n" + "\n".join(f"- {n['title']}: {n['text']}" for n in notes) + f"\n\nQUESTION: {q}"
        msgs = [{"role": "system", "content": SYSTEM}] + [{"role": h["role"], "content": str(h["content"])[:600]} for h in hist] + [{"role": "user", "content": body}]
        for k in range(2):
            try:
                r = httpx.post(f"{explain.OLLAMA_URL}/api/chat", timeout=90, json=dict(
                    model=model, stream=False, keep_alive="30m", options=dict(temperature=(0.3, 0.0)[k], num_predict=260, seed=5 + k), messages=msgs))
                text = r.json()["message"]["content"].strip()
            except Exception as e:
                reason = f"model not reached ({type(e).__name__})"
                break
            if "OUT_OF_SCOPE" in text:
                return done(OUT_TEXT, "outofscope", (), "model judged out of scope")
            ok, reason = self.check_free(text, q, notes)
            if ok:
                return done(text, "llm", notes, model)
        specific = self.topic_stems(q) - WIDE
        if notes[0]["id"] == "plain_summary":
            return done(best, "note", notes[:1], reason)
        if reason.startswith("not grounded") and not (specific & self.topic_stems(" ".join(n["title"] + " " + n["text"] + " " + KEYWORDS.get(n["id"], "") for n in notes))):   # the model answered from general knowledge and no note is a close match
            return done(UNKNOWN_TEXT, "unknown", (), reason)
        return done(best, "note", notes[:1], reason)

    def grounded(self, ans, question, notes):
        """Share of the answer's content words (as stems) that appear in the notes or the question: a general-knowledge answer scores low."""
        have = self.topic_stems(question + " " + " ".join(n["title"] + " " + n["text"] for n in notes)) | self.topic_stems(" ".join(FRIENDLY)) | COMMON
        w = self.topic_stems(ans)
        return len(w & have) / len(w) if w else 1.0

    def check_free(self, ans, question, notes):
        """Light checks for a conversational answer: no new numbers, no banned wording, not empty or rambling. (The sentence-by-sentence
        copy check of check() is what made answers read like the notes, so it is not used here.)"""
        a = ans.strip()
        if not a or len(a.split()) > 170:
            return False, "empty or too long"
        if explain.BANNED.search(re.sub(r"\b(not|no|never|isn't|cannot|can't|nor)\b[^.]{0,40}(diagnos\w+|medical advice|advice|chance of disease|real patients?)|can't help with[^.]*", "", a, flags=re.I)):
            return False, "banned wording"
        if OVERCLAIM.search(a):
            return False, "overclaim"
        if CAUTION.search(question) and re.search(r"\b(does|do|did|will|would) not work\b|\b(doesn't|don't|won't) work\b|\bworks? (well|on real)\b", a, re.I):
            return False, "claims it does or does not work on real patients; the project has not tested that"
        if CAUTION.search(question) and re.match(r"\s*(yes|yeah|yep|sure|absolutely|definitely)\b", a, re.I):
            return False, "says yes to a question about real patients or clinical use"
        allowed = set(self.nums(question))
        for n in notes:
            allowed |= set(self.nums(n["text"] + " " + n["title"]))
        allowed |= {"1", "2", "3", "4", "5"}      # list numbering and small counts
        g = self.grounded(a, question, notes)
        if g < GROUNDED_MIN:
            return False, f"not grounded in the notes ({g:.2f})"
        for n in self.nums(a):
            if n not in allowed:
                return False, f"number {n} is not in the notes"
        return True, "ok"
