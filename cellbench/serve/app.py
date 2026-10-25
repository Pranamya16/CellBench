"""CellBench live server: the interface, the CNN scoring and the Qwen explanations in one small web app.

    uvicorn serve.app:app --app-dir cellbench --port 8000
Qwen runs in Ollama on the same machine (OLLAMA_URL, default http://localhost:11434; in Docker use host.docker.internal).
"""
import json
import os
import re
import sys
import threading
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import httpx
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field

import chat as chatmod
import explain
import generator as G
import upload
import report_layout
import report_pdf
from inference import Scorer

explain.OLLAMA_URL = os.environ.get("OLLAMA_URL", explain.OLLAMA_URL)
explain.MODEL = os.environ.get("QWEN_MODEL", explain.MODEL)
ROOT = G.ROOT
app = FastAPI(title="CellBench live demo")
STATE = {}
REPORTS = {}   # facts_id -> patient + the explanation text shown on the lab report view
FACTS = {}   # facts_id -> facts; the explain endpoint only accepts ids we created, never free text


def results_payload():
    s = pd.read_csv(ROOT / "data" / "results_full_sweep_summary.csv")
    res = {f"{r.noise}|{r.signal_type}|{round(r.affected_fraction * 100)}|{round(r.shift_size * 100)}|{r.model}":
           [round(r.auroc_mean, 3), round(r.auroc_sd, 3), round(r.accuracy_mean, 3)] for r in s.itertuples()}
    dl = pd.read_csv(ROOT / "data" / "detection_limits_all.csv")
    lim = {f"{r.noise}|{r.type}|{r.drop.rstrip('%')}|{r.view}": r.limit for r in dl.itertuples()}
    return res, lim


def limits_for(p):
    """Detection limits (from the sweep) for this patient's scenario, as extra facts for the explanation."""
    if not p["signal"] or p["kind"] == "shift":
        return {}
    _, lim = STATE["results"]
    d = 10 if p["kind"] == "spread" else round(p["drop"] * 100)
    cells = lim.get(f"{p['noise']}|{p['kind']}|{d}|Full per-cell maps, CNN", "")
    rep = lim.get(f"{p['noise']}|{p['kind']}|{d}|Report summary, CNN", "")
    out = {}
    if cells:
        out["sweep_limit_full_cell_data"] = cells if cells == "not reached" else f"{cells} of neutrophils changed"
    if rep:
        out["sweep_limit_report_averages"] = rep if rep == "not reached" else f"{rep} of neutrophils changed"
    return out


@app.on_event("startup")
def startup():
    t = time.time()
    STATE["scorer"] = Scorer()
    STATE["results"] = results_payload()
    STATE["chat"] = chatmod.Chat()
    threading.Thread(target=STATE["chat"].warm, daemon=True).start()
    STATE["boot_s"] = round(time.time() - t, 1)


class PatientRequest(BaseModel):
    kind: str = Field("subgroup", pattern="^(subgroup|spread|shift)$")
    frac: float = Field(0.10, ge=0.0, le=0.30)
    drop: float = Field(0.10, ge=0.0, le=0.30)
    noise: str = Field("shared", pattern="^(shared|independent)$")
    seed: int = Field(1, ge=0, le=100000)
    signal: bool = True


@app.get("/api/health")
def health():
    ollama = False
    try:
        tags = httpx.get(f"{explain.OLLAMA_URL}/api/tags", timeout=3).json()
        ollama = any(m["name"] == explain.MODEL for m in tags.get("models", []))
    except Exception:
        pass
    return dict(ok="scorer" in STATE, boot_s=STATE.get("boot_s"), qwen_ready=ollama, qwen_model=explain.MODEL)


@app.post("/api/patient")
def patient(req: PatientRequest):
    if req.signal and req.kind != "shift" and req.frac <= 0:
        raise HTTPException(400, "share of neutrophils changed must be above 0 for a signal patient")
    t = time.time()
    p = STATE["scorer"].package(req.kind, req.frac, req.drop, req.noise, req.seed, req.signal)
    facts = explain.make_facts(p, limits_for(p))
    fid = uuid.uuid4().hex[:12]
    FACTS[fid] = facts
    REPORTS[fid] = dict(patient=p, text=explain.template(facts), source="template")
    if len(FACTS) > 500:
        old = next(iter(FACTS))
        FACTS.pop(old), REPORTS.pop(old, None)
    p["facts_id"] = fid
    p["ms"] = int((time.time() - t) * 1000)
    return p


class UploadRequest(BaseModel):
    filename: str = Field("cells.csv", max_length=200)
    csv: str = Field(..., max_length=8_000_000)


@app.post("/api/upload")
def upload_endpoint(req: UploadRequest):
    """Score one patient's cell file. Processed in memory and not stored."""
    try:
        cells, warns = upload.parse_cells(req.csv)
    except ValueError as e:
        raise HTTPException(400, str(e))
    t = time.time()
    p = STATE["scorer"].package_cells(cells, upload.safe_name(req.filename), warns)
    facts = explain.make_facts(p, {})
    fid = uuid.uuid4().hex[:12]
    FACTS[fid] = facts
    REPORTS[fid] = dict(patient=p, text=explain.template(facts), source="template")
    if len(FACTS) > 500:
        old = next(iter(FACTS))
        FACTS.pop(old), REPORTS.pop(old, None)
    p["facts_id"] = fid
    p["ms"] = int((time.time() - t) * 1000)
    return p


@app.get("/api/example.csv", response_class=PlainTextResponse)
def example_csv(kind: str = "subgroup", frac: float = 0.10, drop: float = 0.10, noise: str = "shared", seed: int = 1, signal: bool = True):
    """A synthetic patient in the upload format, so the whole loop can be tried."""
    req = PatientRequest(kind=kind, frac=frac, drop=drop, noise=noise, seed=seed, signal=signal)
    cells, _, _ = STATE["scorer"].make_patient(req.kind, req.frac, req.drop, req.noise, req.seed, req.signal)
    body = "SSC,SFL,FSC,FSCW\n" + "\n".join(",".join(str(int(v)) for v in row) for row in cells) + "\n"
    return PlainTextResponse(body, headers={"Content-Disposition": f'attachment; filename="synthetic_patient_{seed}.csv"'})


class ChatRequest(BaseModel):
    message: str = Field(..., max_length=600)
    facts_id: str = ""
    history: list = Field(default_factory=list, max_length=12)   # earlier turns [{role, content}], so follow-up questions work


@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    return STATE["chat"].answer(req.message, FACTS.get(req.facts_id), history=[h for h in req.history if isinstance(h, dict)])


@app.get("/chat", response_class=HTMLResponse)
def chat_page():
    live = (HERE / "live_template.html").read_text()
    style = live[live.index("<style>"):live.index("</style>") + 8]
    return (HERE / "chat_template.html").read_text().replace("__STYLE__", style)


class ExplainRequest(BaseModel):
    facts_id: str


@app.post("/api/explain")
def explain_endpoint(req: ExplainRequest):
    f = FACTS.get(req.facts_id)
    if f is None:
        raise HTTPException(404, "unknown patient; generate it again")
    e = explain.explain(f)
    if req.facts_id in REPORTS:
        REPORTS[req.facts_id].update(text=e["text"], source=e["source"])
    return e


def _layout(fid):
    r = REPORTS.get(fid)
    if r is None:
        raise HTTPException(404, "unknown patient; generate it again")
    return report_layout.build(dict(r, stamp=time.strftime("%d/%m/%Y %H:%M:%S")))


@app.get("/api/report/{fid}")
def report_layout_json(fid: str):
    return _layout(fid)


@app.get("/api/report/{fid}/pdf")
def report_pdf_file(fid: str):
    return Response(report_pdf.render(_layout(fid)), media_type="application/pdf",
                    headers={"Content-Disposition": f'inline; filename="cellbench_synthetic_result_{fid}.pdf"'})


@app.get("/api/results")
def results():
    res, lim = STATE["results"]
    return dict(res=res, lim=lim)


@app.get("/", response_class=HTMLResponse)
def index():
    res, lim = STATE["results"]
    page = (HERE / "live_template.html").read_text()
    return page.replace("__RES__", json.dumps(res)).replace("__LIM__", json.dumps(lim))
