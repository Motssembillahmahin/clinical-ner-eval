# Data, Examples and What the Model Does

## What is the model actually doing?

The model reads a sentence and tags every word with a label:

```
Input words:   The   patient  was  diagnosed  with  Hodgkin   lymphoma  and  treated  with  cyclophosphamide
Output labels:  O      O       O       O        O    B-Disease  I-Disease  O      O       O     B-Chemical
```

That's it. Each word gets one of five labels:

| Label | Meaning |
|---|---|
| `O` | Not an entity (Outside) |
| `B-Disease` | **B**eginning of a disease mention |
| `I-Disease` | **I**nside (continuation of) a disease mention |
| `B-Chemical` | Beginning of a chemical/drug mention |
| `I-Chemical` | Continuation of a chemical/drug mention |

Multi-word entities use B for the first word, I for every word after:
- `"Hodgkin lymphoma"` → `B-Disease I-Disease`
- `"acute respiratory distress syndrome"` → `B-Disease I-Disease I-Disease I-Disease`
- `"metformin"` → `B-Chemical` (single word, no I needed)

---

## Concrete examples

### Example 1 — Disease detection

**Input sentence:**
> *Mutations in the BRCA1 gene are strongly associated with hereditary breast and ovarian cancer.*

**Model output:**

| Word | Label |
|---|---|
| Mutations | O |
| in | O |
| the | O |
| BRCA1 | O |
| gene | O |
| are | O |
| strongly | O |
| associated | O |
| with | O |
| hereditary | B-Disease |
| breast | I-Disease |
| and | I-Disease |
| ovarian | I-Disease |
| cancer | I-Disease |

Extracted entity: **"hereditary breast and ovarian cancer"** → Disease

---

### Example 2 — Chemical detection (drug name)

**Input sentence:**
> *Patients receiving cyclophosphamide showed higher rates of nausea than those on placebo.*

**Model output:**

| Word | Label |
|---|---|
| Patients | O |
| receiving | O |
| cyclophosphamide | B-Chemical |
| showed | O |
| higher | O |
| rates | O |
| of | O |
| nausea | B-Disease |
| than | O |
| those | O |
| on | O |
| placebo | O |

Extracted: **"cyclophosphamide"** → Chemical, **"nausea"** → Disease

---

### Example 3 — Multiple entities in one sentence

**Input sentence:**
> *The combination of methotrexate and leflunomide is used to treat rheumatoid arthritis.*

**Model output:**

| Word | Label |
|---|---|
| The | O |
| combination | O |
| of | O |
| methotrexate | B-Chemical |
| and | O |
| leflunomide | B-Chemical |
| is | O |
| used | O |
| to | O |
| treat | O |
| rheumatoid | B-Disease |
| arthritis | I-Disease |

Extracted:
- **"methotrexate"** → Chemical
- **"leflunomide"** → Chemical
- **"rheumatoid arthritis"** → Disease

---

### Example 4 — What a wrong prediction looks like

The model sometimes makes mistakes. Common error types:

**Boundary error** (gets the entity but wrong span):
> Gold: `"chronic renal failure"` (3 tokens)
> Predicted: `"renal failure"` (missed "chronic")
> Result: 0 entity-level F1 for this instance — the span must match exactly

**False positive** (tags something that isn't an entity):
> Sentence: *"The patient had a family history of cancer."*
> Gold: `B-Disease` on "cancer"
> Predicted might also tag "history" as B-Disease if the model over-fires

**Missing entity** (false negative):
> A complex multi-word disease name like "autosomal dominant polycystic kidney disease" might get partially tagged or missed entirely

---

## Training data

### Datasets used

| Dataset | Source | Entity types | Sentences (train) | Sentences (test) |
|---|---|---|---|---|
| **NCBI Disease** | PubMed abstracts | Disease only | ~593 | ~100 |
| **BC5CDR** | BioCreative literature | Disease + Chemical | ~500 | ~500 |
| **Combined** | | Disease + Chemical | **~1,093** | **~600** |

These are document-level counts — each document (abstract or article) contains multiple sentences.

### Exact split sizes (as seen during training)

```
train:      1,092 examples (sentences)
validation:   500 examples
test:         ~600 sentences → 1,363 labeled entities
                               (707 Disease + 656 Chemical)
```

### What the training data looks like

Each training example is a sentence with word-level BIO labels, for example:

```json
{
  "tokens": ["Familial", "hypercholesterolemia", "is", "treated", "with", "statins"],
  "labels": ["B-Disease", "I-Disease", "O", "O", "O", "B-Chemical"]
}
```

The datasets are downloaded automatically from HuggingFace Hub at training time — nothing needs to be staged locally.

---

## Before vs after training (why fine-tuning matters)

### Before fine-tuning (zero-shot)

The pre-trained BERT models were trained on general text (Wikipedia, PubMed, clinical notes) using a **masked language modeling** objective — predicting randomly hidden words. They have no concept of "Disease" or "Chemical" labels.

When we add the classification head (a single linear layer on top of BERT) for fine-tuning, that head is **randomly initialized**. Without training, it outputs random label distributions. Entity-level F1 before training is effectively ~0% — no meaningful entity spans are correctly identified.

### After 3 epochs of fine-tuning

| Model | Micro-avg F1 |
|---|---|
| Bio_ClinicalBERT | **0.787** (from ~0%) |
| PubMedBERT | **0.819** (from ~0%) |
| BioBERT | **0.833** (from ~0%) |

The fine-tuning teaches the model:
1. Which surface patterns correspond to disease/chemical names
2. Where entity spans begin and end (B vs I distinction)
3. That words like "syndrome", "disease", "cancer" are strong Disease signals
4. That drug names (often long, Latinate, ending in -ine/-mab/-nib) are Chemical signals

### Why don't we report zero-shot numbers?

Zero-shot BERT NER is not a meaningful baseline because the classification head is random — it would be like measuring accuracy before any learning has happened, which is uninformative. The relevant comparison is which *fine-tuned* model performs best, since all three start from effectively the same zero-shot baseline of ~0% entity F1.

If you want a meaningful "no training" baseline, the common approach is to use a few-shot or prompt-based model (like GPT-4 with entity extraction prompts). We did not run that comparison here.

---

## How the data flows through the pipeline

```
Raw sentence text
        │
        ▼
Whitespace tokenization         ("rheumatoid arthritis" → ["rheumatoid", "arthritis"])
        │
        ▼
BERT subword tokenization       ("arthritis" → ["arth", "##ritis"])
        │
        ▼
Label alignment                 (assign B-Disease to "arth", mark "##ritis" as -100 / ignored)
        │
        ▼
BERT forward pass               (contextual embeddings for each token)
        │
        ▼
Linear classification head      (5-class softmax: O / B-Disease / I-Disease / B-Chemical / I-Chemical)
        │
        ▼
Argmax prediction per token
        │
        ▼
Collapse subwords back to words (take label of first subword, discard ##-continuations)
        │
        ▼
seqeval entity-level scoring    (compare predicted spans to gold spans)
```

---

## Why entity-level scoring is strict

seqeval requires the full entity span to match. Partial credit is not given:

| Gold | Predicted | Score |
|---|---|---|
| `B-Disease I-Disease` ("renal failure") | `B-Disease I-Disease` | ✓ True positive |
| `B-Disease I-Disease` ("renal failure") | `B-Disease O` ("renal" only) | ✗ False negative (missed) + False positive (wrong span) |
| `O O` (no entity) | `B-Disease I-Disease` | ✗ False positive |

This strictness is why our F1 scores (0.79–0.83) represent genuinely strong performance — getting the exact boundary of multi-word disease names consistently is hard.
