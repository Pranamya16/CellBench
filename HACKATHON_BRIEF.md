# Leeds AI Society Hackathon: AI in Healthcare (24-25 Oct 2026)

Text extracted from photos of the event slides. Anything cut off in a photo is marked [cut off].

## 1. The challenge ("What will you build?")

An open theme at the intersection of AI and healthcare.

1. **Find a need:** choose a problem worth solving for patients, clinicians or researchers.
2. **Build an idea:** use data, AI and software thoughtfully. Get something working.
3. **Demo impact:** show what it does, who it helps and where the limitations are.

Examples on the slide:
- A plain-English explainer for discharge letters
- A missed-appointment predictor trained on synthetic data
- A triage assistant for minor [cut off]
- A medication reminder designed for older patients
- A skin-lesion classifier built on public image datasets

## 2. Ground rules (short version; full rules are on Devpost)

1. **Build it here:** templates and ideas.
2. **Disclose everything:** pre-existing work and AI tools are fine if declared. Hidden work gets removed.
3. **No real patient data:** synthetic or open public datasets only, even if anonymised. Example: Kaggle, [cut off].
4. **Look out for each other:** MLH Code of Conduct applies. Anything wrong? Tell any committee member.

## 3. How we'll be judged

Four shared criteria, plus a separate Responsible Data award.

1. **Creativity & innovation:** how fresh and thoughtful is the concept?
2. **Technical sophistication:** how strong is the implementation?
3. **Impact:** could it meaningfully help people?
4. **Design & user experience:** is it clear and usable?
5. **Most responsible use of data (separate award):** is the data appropriate, and have privacy, consent and bias been considered? Explain your choices in the data question in the submission [partly cut off].

## 4. OpenAI prototyping exercise (slides 11 and following)

**Choose your first idea.** Turn and talk with your group about the product prototype you want to create.

Consider:
1. Who is the product for?
2. What problem should it solve first?
3. What should someone be able to click, enter, or complete?

Complete the sentence: "I'm prototyping ______ for ______ so they can ______."

**Discuss with your group:**
1. What already works?
2. What feels generic or incomplete?
3. What is one improvement you want to make?

## 5. Guest talk slide: the surgical patient pathway

| Stage | What happens | AI ideas on the slide |
| --- | --- | --- |
| Symptoms | Assessment in primary care; referral for imaging or surgical opinion | LLM for symptoms to diagnosis; LLM for streamlined referral; CNN for imaging diagnosis |
| Surgical assessment + decision for surgery | Further imaging + tests; +/- MDT discussion | Multi-modal AI for MDT diagnosis; 3D modelling to enhance MDT; surgical risk prediction tool |
| Surgery | Admission; consent; anaesthesia; surgical procedure | LLM to enhance consent; 3D modelling for surgical navigation; VLM for decision-support |
| Post-op | Daily reviews; blood tests +/- imaging; wound care | Post-op prediction tool; VLM for decision-support |
| Follow-up | Clinic reviews; histology; imaging | Automated follow-up; CNN for histology/imaging analysis |

## 6. Housekeeping

- Two QR codes on screen: **Devpost** and **WhatsApp** (the links cannot be read from the photo).
- Event: AI in Healthcare, 24-25 Oct, 2026 / HELIX. Room shown in the background: HELIX Classroom (7.14).
- Leeds AI Society banner: Talks (industry speakers), Workshops (hands-on sessions), Projects (build with others), Hackathons (compete with others). Instagram /leedsaisoc, website leedsaisoc.co.uk, LinkedIn /company/leedsaisoc.

## 7. Our project (from chat, so Claude Code has the context)

**Project:** a "detection ruler" for routine blood-test analyser data (Sysmex). The analyser measures tens of thousands of cells per sample, but the report keeps only a few dozen summary numbers.

**Draft sentence:** "I'm prototyping an interactive detection ruler for hospital lab teams and AI researchers, so they can check whether a claimed 'AI finds X in a blood test' signal is big enough to be believable."

**Idea:**
- Generate realistic fake per-cell data for peripheral-blood-like patients, in the Sysmex layout (see DATA_DESIGN.md).
- Plant a fake MDS-style signal: a share of neutrophils made less granular (lower SSC, with SFL and FSC dropping alongside). Its size is set by the **affected fraction** (0, 2, 5, 10, 20%) and the **shift size** (5, 10, 20%).
- Test whether models find it using summary numbers only versus full per-cell data.
- Shrink the affected fraction until they can't. The curve is the ruler.
- *Updated 2026-10-24: the earlier "strength" parameter and the body-fluid, SFL-based signal were replaced. The old version is in `archive/`.*

**Notes for the build:**
- Data must be synthetic or open public only (rule 3). Explain data choices in the submission for the Responsible Data award.
- Declare pre-existing work and that Claude was used (rule 2).
- Seed files: four public FCS files from Ai et al. 2018 (PLOS ONE). Those are body-fluid samples, not whole blood. Our synthetic patients are blood-like (neutrophil-dominant), but the cell positions are borrowed from these body-fluid files and the shares are assumed. State this as a limitation.
- Test spread changes and a separate low-granularity subgroup as well as a mean shift (types `spread`, `subgroup`, `shift`). The Chen et al. 2026 preprint found the strongest signals in cell variance and distribution tails.
- Models to compare: simple baselines, TabPFN on per-sample summary features, a small set-based network trained on the fake cells.
- Do not claim real diseases produce signals this large. The ruler only says how big a signal must be.
- Demo should show what it does, who it helps, and where the limitations are.
