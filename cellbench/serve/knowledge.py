"""The only source the chatbot may answer from: short notes about the project, written from our own files.

Every figure that comes from an experiment is read from the result tables at start-up, so the notes cannot drift from the data.
Notes are deliberately short and plain. If a question is not covered by a note, the chatbot says so.
"""
import pandas as pd

import generator as G

ROOT = G.ROOT


def _limit(dl, noise, kind, drop, view):
    r = dl[(dl.noise == noise) & (dl.type == kind) & (dl["drop"] == drop) & (dl.view == view)]
    return r.limit.iloc[0] if len(r) else None


def _res(s, noise, kind, f, d, model):
    r = s[(s.noise == noise) & (s.signal_type == kind) & (abs(s.affected_fraction * 100 - f) < 1e-6) & (abs(s.shift_size * 100 - d) < 1e-6) & (s.model == model)]
    return float(r.auroc_mean.iloc[0]) if len(r) else None


def build_cards():
    s = pd.read_csv(ROOT / "data" / "results_full_sweep_summary.csv")
    dl = pd.read_csv(ROOT / "data" / "detection_limits_all.csv")
    try:
        fm = pd.read_csv(ROOT / "data" / "final" / "cnn" / "final_metrics.csv")
        ov = fm[fm.signal_type == "CALIBRATION_ERROR_trained_scenarios"].set_index("view")
        final = (f"On brand-new test patients, the final full cell data CNN scored {ov.loc['cells', 'auroc']:.2f} AUROC overall ({ov.loc['cells', 'accuracy'] * 100:.0f}% correct) "
                 f"and the final report averages CNN scored {ov.loc['report', 'auroc']:.2f} ({ov.loc['report', 'accuracy'] * 100:.0f}% correct).")
    except Exception:
        final = "The final models were trained once on mixed scenarios and tested on brand-new synthetic patients."
    cc, cr = "Full per-cell maps, CNN", "Report summary, CNN"
    lim = lambda kind, drop, view, noise="shared": _limit(dl, noise, kind, drop, view)
    a10, a10r = _res(s, "shared", "subgroup", 10, 10, "cells_CNN"), _res(s, "shared", "subgroup", 10, 10, "cnn_report_auto")
    sp, spr = _res(s, "shared", "spread", 10, 10, "cells_CNN"), _res(s, "shared", "spread", 10, 10, "cnn_report_auto")
    cards = [
        ("plain_summary", "The results in plain words",
         "We made up fake patients and secretly changed some of their white blood cells. Then we asked two AI models to spot the change. One model saw only the averages that a normal lab report keeps. The other saw every single cell. "
         f"When only a small share of cells was changed, the averages hid it, but the model that saw every cell still found it. For example, when 10% of one cell type was changed, the every-cell model scored {a10:.2f} and the averages model scored {a10r:.2f}, where 1.0 is perfect and 0.5 is a coin flip. "
         "When the change was very tiny, neither model could see it. All of this is on made-up data, so it does not show that the tool works on real patients."),
        ("about", "What CellBench is",
         "CellBench is a research demo that measures how small a change in blood cells an AI can still see. It plants a known change in some neutrophils of made-up (synthetic) patients and tests whether a model that sees only report averages, or a model that sees every cell, can find it."),
        ("headline", "The main result",
         f"With the same type of model (a CNN), full cell data detected a change in {lim('subgroup', '20%', cc)} of neutrophils at a 20% side-scatter drop, while report averages needed {lim('subgroup', '20%', cr)}. "
         f"At a 10% drop and 10% of neutrophils changed, the full cell CNN scored {a10:.2f} and the report averages CNN scored {a10r:.2f}. These are results on synthetic data."),
        ("limits_subgroup", "Detection limits for a group of changed neutrophils",
         f"At a 20% side-scatter drop the full cell CNN detects the change from {lim('subgroup', '20%', cc)} of neutrophils and report averages from {lim('subgroup', '20%', cr)}. "
         f"At a 10% drop the full cell CNN needs {lim('subgroup', '10%', cc)} and report averages never reach the detection line. With independent noise the full cell CNN needs {lim('subgroup', '20%', cc, 'independent')} at a 20% drop."),
        ("spread", "Spread change",
         f"When the neutrophil cluster only widens and its centre does not move, report averages cannot see it (score {spr:.2f}, about a coin flip), while the full cell CNN scores {sp:.2f}. Averages miss it because the average does not change."),
        ("shift", "When every neutrophil shifts",
         "If every neutrophil shifts down by 10%, both the full cell data and the report averages see it with a score near 1.0. That is the case averages are built for. The advantage of full cell data appears when only some neutrophils change, or when the spread changes."),
        ("five_percent", "Why a 5% drop is not detected",
         "A 5% drop in side scatter is about the same size as the natural variation between patients (about 5%), so no model we tested could detect it at any share of neutrophils changed. This is a limit of the data, not of one model."),
        ("detection_limit", "What the detection limit means",
         "The detection limit is the smallest share of neutrophils changed whose average score is at least 0.80 and stays at or above 0.80 for every larger share we tested. We defined it before looking at the results. If no tested share reaches it, the limit is reported as not reached."),
        ("auroc", "What AUROC means",
         "AUROC is a score from 0.5 to 1.0 for how well a model separates normal patients from changed patients. 0.5 is a coin flip and 1.0 is perfect. Pick one changed and one normal patient at random: the AUROC is how often the model ranks the changed one as more abnormal. Differences under about 0.05 are within noise."),
        ("accuracy", "Accuracy",
         "Plain accuracy is the share of patients classified correctly. We mostly report AUROC because it does not depend on a cutoff. Where we show accuracy, the cutoff was chosen on other patients, not on the ones being scored."),
        ("detector_score", "What the detector score is",
         "The detector score runs from 0 to 100 and says how strongly a patient looks like the planted pattern to a model. It is calibrated on a 50/50 mix of made-up normal and changed patients. It is not a chance of disease and not a diagnosis."),
        ("report_view", "The report averages view",
         "The report averages view imitates what a routine lab report keeps: for each cell group a few averages and shares, such as the neutrophil side scatter average. It is a stand-in built from simulated cells, not a real lab report."),
        ("cell_view", "The full cell data view",
         "The full cell data view gives the model every cell, as three count maps (side scatter against side fluorescence, side scatter against forward scatter, and forward scatter against side fluorescence). The model gets no group labels."),
        ("gate", "The automatic gate",
         "A gate assigns each cell to a group, such as neutrophil or lymphocyte, using only its four readings. We trained a simple one on separate reference patients. It is imperfect: it often confuses eosinophils and immature cells with neutrophils, so those lines are approximate."),
        ("readings", "The four readings",
         "Each cell has four readings on a scale of 0 to 255. SSC is side scatter, which reflects how grainy a cell is inside. SFL is side fluorescence, which reflects the dye signal and nucleic acid content. FSC is forward scatter, which reflects size. FSCW is the width of the forward scatter pulse."),
        ("hypogranular", "Hypogranular neutrophils",
         "Hypogranular neutrophils have fewer granules, so they scatter less light sideways and show a lower side scatter. We plant this change in some neutrophils as our test signal. In real life it can occur in conditions such as myelodysplastic syndromes, but we do not show that any real disease produces changes of these sizes."),
        ("signal_types", "The three kinds of change we plant",
         "Subgroup: a share of neutrophils becomes less granular. Shift: every neutrophil becomes less granular. Spread: the neutrophil cluster becomes wider without moving. The size of the change is the side scatter drop (5%, 10% or 20%)."),
        ("data_made", "How the synthetic data was made",
         "Each patient has about 2,000 cells drawn from six groups (neutrophils, lymphocytes, monocytes, eosinophils, immature granulocytes and debris), with a different cell mix and noise for every patient. A cohort has 300 normal and 300 changed patients. Everything is rebuilt exactly from a seed."),
        ("borrowed", "Where the cell positions come from",
         "The positions of the cell groups were measured from four public body-fluid files (Ai et al. 2018, PLOS ONE) and borrowed for a blood-like mix. The cell mix, the noise size, the change sizes and the linked drop in the other readings are assumptions. They should be re-measured on real blood files."),
        ("noise", "Shared and independent noise",
         "Patients differ from each other. With shared noise one patient-level factor moves all cell groups together, so a model can cancel it by comparing groups. With independent noise each group varies on its own, which is harder. Independent noise hurt the report averages most; the full cell CNN lost much less."),
        ("models", "The models",
         "We trained the same two model types on both data views: a small CNN and a small MLP. The report averages CNN reads the report numbers laid out as a grid. Models were trained on Kaggle. Patients are split by patient, never by cell, and the sweep used 5 seeds."),
        ("checks", "How we checked honesty",
         "With no change planted every model scored about 0.5, so nothing leaks the answer. Tuning used different seeds from the main sweep. A shortcut check added extra cells to the empty map area and the cell map scores dropped only slightly."),
        ("final_models", "The final models", final + " They were trained on fresh seeds, calibrated, and are the models used in the live demo."),
        ("live_demo", "The live demo",
         "The live page makes a brand-new synthetic patient, scores it with the final CNNs, and shows the cell map, a lab-style result sheet and a plain-language explanation. The heatmap shows where the model looked, not why a patient would be ill."),
        ("upload", "Uploading a file",
         "You can upload one patient's cell file as a CSV with the columns SSC, SFL, FSC and FSCW (0 to 255, at least 500 cells). It is processed in memory and not stored. Use synthetic or open data only, never real patient data. The models know only simulated patients, so scores for other data are not meaningful."),
        ("qwen", "What Qwen does here",
         "Qwen 2.5 (0.5B), running locally, rewrites a correct draft explanation in simpler words. Every rewrite is checked: numbers must match, it must not contradict the scores or add new content, and if it fails the draft is shown instead. The chat page uses a larger Qwen 2.5 (3B) with a strict prompt: it may only use the project notes."),
        ("not_diagnostic", "Not a diagnosis",
         "CellBench is not a diagnostic tool and gives no medical advice. Scores are detector scores on synthetic data. We do not claim any real disease produces changes of these sizes, and a score is never a chance of disease."),
        ("limits", "Main limitations",
         "All data is synthetic. Positions are borrowed from body-fluid files. Several settings are assumptions. Our fake normal cells leave the key map area nearly empty, which may make the cell map CNN look better than it would on real data. The report averages view uses an automatic gate that is not exactly reproducible across machines."),
        ("future", "What would come next",
         "Next steps would be re-measuring cell positions on real blood files from a lab analyser, testing on real data with ethics approval, and checking whether real conditions produce changes large enough to detect. Only then could any clinical use be considered."),
        ("ai_tools", "AI tools used",
         "The code, analysis and documents were written with Claude Code (Anthropic). The plain-words explanation of each scored patient uses Qwen 2.5 (Alibaba) running locally through Ollama; this chat shows our own notes word for word. The models were trained on Kaggle."),
        ("real_data", "Is the data real?",
         "No. All patients are synthetic (made up). No real patient data is used. Only the positions of the cell groups were measured from four public body-fluid files, and then borrowed. CellBench has not been tested on real patients, and its models know only simulated patients."),
        ("neutrophil", "What a neutrophil is",
         "Neutrophils are the most common type of white blood cell and help the body fight infection. In this project they are the cells where we plant a change, because their side scatter reflects how many granules they have."),
        ("smallest_tested", "The smallest change we tested",
         "The smallest share of changed neutrophils we tested was 2%, and the smallest drop in side scatter was 5%. Anything smaller, such as 1% of neutrophils, was not tested, so we cannot say whether it could be detected."),
        ("not_validated", "Has it been validated?",
         "No. CellBench has not been validated clinically or on real patients. Everything was tested on synthetic patients that we made. Validation would need real blood data from a lab analyser and ethics approval."),
        ("hackathon", "The hackathon",
         "CellBench was built for the Leeds AI Society AI in Healthcare hackathon on 24 and 25 October 2026. The live demo is on Sunday 25 October."),
        ("analyser", "The blood analyser and the data",
         "Sysmex XN is a blood analyser that measures blood cells one by one with a laser and a fluorescent dye. For each cell it records side scatter, side fluorescence and forward scatter, but a routine report keeps only averages and counts. CellBench asks how much the averages hide."),
        ("cnn_mlp", "CNN and MLP",
         "A CNN (convolutional neural network) is an AI model that scans small patches of an image or grid, so it can pick up local patterns. An MLP is a simpler network that reads a flat list of numbers. We tested both types on both views so the comparison is fair."),
        ("cnn_detail", "How the CNNs are built",
         "The cell map CNN has three small convolution layers (16, 32 and 32 filters) that read the three 32 by 32 count maps, followed by one output. The report averages CNN is a smaller network that reads the 6 by 5 grid of report numbers. Each model is trained three times from different random starts, the three are averaged, and training stops early using a separate validation split."),
        ("calibration", "Calibration",
         "Calibration turns the raw model output into the 0 to 100 detector score. It is fitted on separate validation patients with a 50/50 mix of normal and changed patients, so the score is a detector score in that mix and not a chance of disease."),
        ("seed", "Seeds",
         "A seed is a number that makes random choices repeatable: the same seed rebuilds exactly the same synthetic patients. The main sweep used 5 seeds, each a fresh set of patients, so the results are averages over 5 repeats and we can see how much they vary."),
        ("cohort", "Cohorts",
         "A cohort is one set of synthetic patients made with one setting: 300 normal and 300 changed patients, with about 2,000 cells each. The models are tested on patients they never saw during training, with patients split by patient and never by cell."),
        ("heatmap", "The heatmap",
         "The heatmap on the live page shows where on the side scatter against side fluorescence map the cell map CNN looked most when it scored the patient. It shows where the model looked, not why a patient would be ill."),
        ("lab_sheet", "The lab result sheet",
         "The lab result sheet is a mock-up of an analyser result page for the patient on screen, in the layout and font of a standard result sheet. It shows a differential from the automatic gate, three neutrophil averages, the two detector scores and a plain-words note, and it can be downloaded as a PDF. The white cell total is made up. It says on the sheet that the data is simulated and a proof of concept."),
        ("speed", "How fast it is",
         "Scoring one patient with the final CNNs takes about 50 milliseconds on the demo laptop. Qwen then writes the plain-words explanation, which takes a little longer."),
        ("run_demo", "How to run the demo",
         "Install Docker Desktop and Ollama, run ollama pull qwen2.5:0.5b once, then run ./run_demo.sh and open http://localhost:8000. The chat is on the page /chat. Stop it with docker stop cellbench-live."),
        ("docker", "Docker",
         "Docker packages the live demo (the page, the scoring models and their code) into one container so it runs the same way on any laptop. Qwen runs in Ollama outside the container on the same machine."),
        ("kaggle", "Where the models were trained",
         "The model training and the sweeps ran on Kaggle, not on a laptop: the CNNs on free T4 GPUs and the MLPs on CPU. The live demo only scores patients with the saved models."),
        ("chatbot", "This chat",
         "This chat is written by Qwen 2.5 (3B) running locally. It looks up the most relevant project notes by meaning, answers only from them, shows which notes it used, remembers the last few questions for follow-ups, and refuses medical questions and anything outside the project."),
        ("mds", "What MDS is",
         "MDS (myelodysplastic syndromes) is a group of blood conditions in which blood cells develop abnormally. One feature that can occur is neutrophils with fewer granules. We use it only as the motivation for the planted change. CellBench does not diagnose MDS or any other disease."),
        ("white_cells", "The white blood cell groups",
         "We simulate five kinds of white cell plus debris. Neutrophils are the most common and fight infection. Lymphocytes are immune cells. Monocytes are larger cells that clear up debris and germs. Eosinophils take part in allergic responses. Immature granulocytes are young cells that are normally rare in blood."),
        ("rich", "Richer summaries",
         "We also tried a richer summary of each cell group (percentiles and spread). It recovered part of the signal, but it was given or estimated the cell groups, which the full cell maps do not need. It is a footnote to the main comparison."),
    ]
    return [dict(id=i, title=t, text=x) for i, t, x in cards]


def patient_card(facts):
    """A note about the patient the user just scored (built from the same facts the explanation uses)."""
    if not facts:
        return None
    f = {k: v for k, v in facts.items() if k != "allowed_numbers"}
    text = (f"The patient you just scored: {f['change']}. The full cell data model scored it {f['score_full_cell_data']} out of 100 and {f['cells_verdict']}; "
            f"the report averages model scored it {f['score_report_averages']} out of 100 and {f['report_verdict']}.")
    return dict(id="patient", title="The patient you just scored", text=text)
