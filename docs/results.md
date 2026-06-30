# Results and Analysis

## Summary

All three models were fine-tuned for 3 epochs on the combined NCBI Disease + BC5CDR dataset and evaluated on the combined test set (1,363 entities). Scores are entity-level F1 from seqeval.

---

## Full results table

### Micro-average F1 (primary metric)

| Model | Precision | Recall | F1 |
|---|---|---|---|
| **BioBERT** | **0.8133** | **0.8533** | **0.8328** |
| PubMedBERT | 0.8032 | 0.8357 | 0.8191 |
| Bio_ClinicalBERT | 0.7622 | 0.8136 | 0.7871 |

### By entity type

**Chemical** (656 test entities):

| Model | Precision | Recall | F1 |
|---|---|---|---|
| PubMedBERT | 0.8492 | 0.8841 | 0.8663 |
| BioBERT | 0.8591 | 0.8826 | 0.8707 |
| Bio_ClinicalBERT | 0.8443 | 0.8598 | 0.8520 |

**Disease** (707 test entities):

| Model | Precision | Recall | F1 |
|---|---|---|---|
| BioBERT | 0.7725 | 0.8260 | 0.7984 |
| PubMedBERT | 0.7605 | 0.7907 | 0.7753 |
| Bio_ClinicalBERT | 0.6925 | 0.7709 | 0.7296 |

Full precision scores are in `results/full_report.json`.

---

## Analysis

### 1. BioBERT wins overall, but PubMedBERT is competitive

BioBERT leads on micro-average F1 (0.833 vs 0.819), but the gap is only 1.4 points — within the range where a different random seed or dataset split might flip them. Both are strong choices. PubMedBERT actually edges BioBERT slightly on Chemical F1 (0.866 vs 0.871 — too close to call).

### 2. Chemical is easier than Disease for all models

Every model scores ~5–8 F1 points higher on Chemical than on Disease:

| Model | Chemical F1 | Disease F1 | Gap |
|---|---|---|---|
| BioBERT | 0.871 | 0.798 | 7.3 |
| PubMedBERT | 0.866 | 0.775 | 9.1 |
| Bio_ClinicalBERT | 0.852 | 0.730 | 12.2 |

**Why?** Chemical names (drug names, compounds) tend to be distinctive surface forms that rarely appear outside their entity class. Disease names are more ambiguous — "failure", "syndrome", "infection" are common English words that only become disease names in certain contexts. Boundary detection is harder too: "acute respiratory distress syndrome" is one disease entity spanning five tokens.

### 3. Bio_ClinicalBERT underperforms despite the domain match

Bio_ClinicalBERT was trained on MIMIC-III clinical notes, which seems like the best domain fit. But it scores 4–5 F1 points below the other two. The likely explanation:

- **Training data mismatch**: NCBI Disease and BC5CDR are curated from PubMed abstracts and biomedical literature — closer to BioBERT's and PubMedBERT's pre-training domain than to raw clinical notes.
- **Annotation style**: Clinical notes use abbreviations, shorthand, and implicit entities ("pt c/o SOB" = "patient complains of shortness of breath") that don't appear in the structured NCBI/BC5CDR annotations.
- **Tokenization**: Clinical abbreviations and drug names written as clinicians write them may not align well with the vocabulary learned from MIMIC.

In short: Bio_ClinicalBERT's advantage kicks in when the *test data* is also clinical notes. On literature-derived benchmarks it is at a disadvantage.

### 4. Recall is consistently higher than precision

All models find more entities than they should (higher recall) rather than missing too many (lower precision). This means the models err on the side of over-tagging. In a clinical pipeline this is often acceptable — it is better to flag a non-entity for human review than to miss a real disease mention.

### 5. Training loss trends were predictive

BioBERT and PubMedBERT both converged faster and to a lower final loss:

| Model | Final train loss | Micro F1 |
|---|---|---|
| BioBERT | 0.227 | 0.833 |
| PubMedBERT | 0.225 | 0.819 |
| Bio_ClinicalBERT | 0.261 | 0.787 |

The ~15% higher final loss for Bio_ClinicalBERT directly maps to its lower F1.

---

## Limitations

- **Benchmark domain**: Both datasets (NCBI Disease, BC5CDR) come from biomedical literature abstracts, not clinical notes. Results may shift significantly on actual clinical text — Bio_ClinicalBERT may close the gap or overtake on real doctor notes.

- **Only 3 epochs**: Standard for fine-tuning, but longer training could change the ordering. Early stopping based on entity-level F1 (rather than eval loss) might also alter results.

- **MACCROBAT excluded**: MACCROBAT (real clinical case reports with 24 entity types) was not available on the HuggingFace Hub when this evaluation ran. Adding it would make the benchmark more representative of clinical text.

- **No ensemble or calibration**: Results reflect single-model fine-tuning. Ensembling or confidence calibration are not explored.

---

## Recommendation

For a production clinical NER pipeline benchmarked on biomedical literature:

- **Use BioBERT** if you want the safest single-model choice — it led on both entity types and overall.
- **Use PubMedBERT** if Chemical recognition is more important than Disease (it nearly ties BioBERT there and may generalise better to new chemical names).
- **Revisit Bio_ClinicalBERT** if your actual inference data is real clinical notes (discharge summaries, progress notes) rather than literature — the domain match may make it competitive or superior in that setting.
