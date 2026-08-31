# A Universal Vibe? Finding and Controlling Language-Agnostic Informal Register with SAEs
**Ayal Klein, Avraham Shtrasberg, Uri Z. Kialy**

This repository contains the code, data, and results for our ACL submission investigating whether multilingual LLMs represent slang as a universal, language-agnostic concept or as a collection of isolated language-specific patterns.

We use Sparse Autoencoders (SAEs) on **Gemma-2-9B-IT** across **English, Hebrew, and Russian**, and show that a small universal core of slang-discriminative features forms a geometrically coherent "slang island" in the model's latent space — one that transfers zero-shot to **German**, a language entirely absent from our feature extraction pipeline.

---

## Key Findings

- **Universal slang core**: 9–10 SAE features are shared across all three languages at both Layer 9 and Layer 20, forming a geometrically coherent cluster in decoder space.
- **Statistically non-accidental**: A permutation test (n=100,000), including a robustness check restricted to activity-filtered features, confirms the triple overlap is a 125,000× enrichment over chance (p < 10⁻⁵).
- **Causal control**: Steering with universal feature vectors causally modulates generation formality across all languages, including zero-shot transfer to 6 languages (German,
Japanese, Hindi, Thai, Georgian, and Amharic).
- **Input/output asymmetry**: SAE-derived vectors control open-ended generation.

---

## Repository Structure

```
.
├── layer20_features_extraction/
│   ├── Layer20_crosslinguistic_analysis.ipynb     # SAE feature extraction & overlap analysis
│   ├── layer20_crosslinguistic_analysis.json      # Top-100 features per language + universal core
│   ├── slang_features_layer20_comprehensive.json  # Full feature metrics
│   └── permutation_test_null.py                   # Statistical significance of feature overlap
│                                                    #   (+ activity-filtered robustness check)
│
├── layer9_steering_generation/
│   ├── Layer9_analysis.ipynb.zip              # Layer 9 feature extraction & steering (zipped notebook)
│   ├── cross_linguistic_steering_notebook_(1).py   # Script export of the Layer 20 steering notebook
│   ├── feature_top_activations.pkl            # Cached top-activating examples per feature
│   └── *.csv / *.png                          # Results, correlations, and plots
│
├── layer 20 cross linguistic steering generations/
│   ├── cross_linguistic_steering_notebook_(1).ipynb  # Layer 20 generation steering
│   ├── Corrected_layer20_run.ipynb                   # Revised steering run (narrower alpha sweep,
│   │                                                  #   GPT-4o-as-judge formality scoring)
│   ├── Correlation_test.py                           # Pearson correlation analysis
│   ├── formality_scores.csv                          # GPT-4o-mini formality scores
│   ├── formality_scores_human&LLM_annotated.csv      # Human + LLM annotation comparison
│   └── *.csv / *.png                                 # Steering heatmaps, transfer matrices,
│                                                       #   and generation results
│
└── slang_feature_analysis_voacb_proj.ipynb   # Vocabulary projection of universal features
```

---

## Datasets

Three datasets of **polysemous** slang/literal sentence pairs — every target term is in-vocabulary, appearing in both a slang and a literal reading within the same dataset.

| Language | Sentences | Slang | Literal | Unique terms | Source |
|----------|-----------|-------|---------|--------------|--------|
| English  | 2,835     | 968   | 1,857   | 130          | OpenSubtitles, web |
| Hebrew   | 4,527     | 1,776 | 2,751   | 18           | HuggingFace corpora |
| Russian  | 1,259     | 538   | 721     | 15           | VKontakte (VK), Telegram |

---

## Reproducing the Results

### 1. Feature Extraction (Layer 20)
Open and run `layer20_features_extraction/Layer20_crosslinguistic_analysis.ipynb`.
Requires: `transformer_lens`, `sae_lens`, access to Gemma-2-9B-IT and GemmaScope SAEs.

### 2. Permutation Test
```bash
python layer20_features_extraction/permutation_test_null.py \
    --json layer20_features_extraction/layer20_crosslinguistic_analysis.json \
    --n_perm 100000 \
    --plot
```
This also runs a robustness check that repeats the permutation test over the activity-filtered feature pool only (≥5% slang activation rate, ≥10 total firings), reported for Layer 9 and Layer 20 separately.

### 3. Generation Steering (Layer 9 & 20)
Unzip and run `layer9_steering_generation/Layer9_analysis.ipynb.zip`, then run the notebooks in `layer 20 cross linguistic steering generations/`. `Corrected_layer20_run.ipynb` is the revised Layer 20 steering run (narrower alpha sweep, GPT-4o-as-judge formality scoring); `cross_linguistic_steering_notebook_(1).ipynb` contains the original run.

### 4. Vocabulary Projection
Run `slang_feature_analysis_voacb_proj.ipynb` to see which tokens the universal features promote.

---

## Requirements

```
torch
transformer_lens
sae_lens
numpy
pandas
matplotlib
scipy
```

All notebooks were developed and tested on **Google Colab** with a T4/A100 GPU.
Model: `google/gemma-2-9b-it`
SAEs: GemmaScope 131k-width at Layer 9 and Layer 20.

---

## Citation

Please cite us :)  :
```
@article{kialy2026universal,
  title={A Universal Vibe? Finding and Controlling Language-Agnostic Informal Register with SAEs},
  author={Kialy, Uri Z and Shtarkberg, Avi and Klein, Ayal},
  journal={arXiv preprint arXiv:2603.26236},
  year={2026}
}
```
