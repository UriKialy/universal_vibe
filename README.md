# A Universal Vibe? Finding and Controlling Language-Agnostic Informal Register with SAEs

This repository contains the code, data, and results for our ACL submission investigating whether multilingual LLMs represent slang as a universal, language-agnostic concept or as a collection of isolated language-specific patterns.

We use Sparse Autoencoders (SAEs) on **Gemma-2-9B-IT** across **English, Hebrew, and Russian**, and show that a small universal core of slang-discriminative features forms a geometrically coherent "slang island" in the model's latent space — one that transfers zero-shot to **German**, a language entirely absent from our feature extraction pipeline.

---

## Key Findings

- **Universal slang core**: 9–10 SAE features are shared across all three languages at both Layer 9 and Layer 20, forming a geometrically coherent cluster in decoder space.
- **Statistically non-accidental**: A permutation test (n=100,000) confirms the triple overlap is a 125,000× enrichment over chance (p < 10⁻⁵).
- **Causal control**: Steering with universal feature vectors causally modulates generation formality across all languages, including zero-shot transfer to German.
- **Input/output asymmetry**: SAE-derived vectors control open-ended generation; Difference-in-Means vectors control classification — consistent with the input/output feature distinction of Arad et al. (2025).

---

## Repository Structure

```
.
├── datasets/
│   ├── unified_english_dataset.csv          # English polysemous slang/literal sentences
│   ├── hebrew_slang_dataset.csv             # Hebrew polysemous slang/literal sentences
│   ├── russian_literal_negatives_dataset.csv # Russian literal sentences
│   └── ...                                  # Classification outputs, human annotations
│
├── layer20_features_extraction/
│   ├── Layer20_crosslinguistic_analysis.ipynb     # SAE feature extraction & overlap analysis
│   ├── layer20_crosslinguistic_analysis.json      # Top-100 features per language + universal core
│   ├── slang_features_layer20_comprehensive.json  # Full feature metrics
│   └── permutation_test.py                        # Statistical significance of feature overlap
│
├── layer9_steering_generation/
│   ├── Layer9_analysis.ipynb               # Layer 9 feature extraction & steering
│   └── *.csv / *.png                       # Results and plots
│
├── layer 20 cross linguistic steering generations/
│   ├── cross_linguistic_steering_notebook_(1).ipynb  # Layer 20 generation steering
│   ├── Correlation_test.py                           # Pearson correlation analysis
│   ├── formality_scores.csv                          # GPT-4o-mini formality scores
│   ├── formality_scores_human&LLM_annotated.csv      # Human + LLM annotation comparison
│   └── *.png                                         # Steering heatmaps and transfer matrices
│
├── layer20_steering_classification_/
│   ├── layer_20_classifcation+steering.ipynb  # DiM steering on classification task
│   ├── alpha_sweep_all_results.csv            # P(slang) across all α values
│   └── baseline_checkpoint.csv               # Unsteered classification baseline
│
└── slang_feature_analysis_voacb_proj.ipynb   # Vocabulary projection of universal features
```

---

## Datasets

Three datasets of **polysemous** slang/literal sentence pairs — every target term is in-vocabulary, appearing in both a slang and a literal reading within the same dataset.

| Language | Sentences | Slang | Literal | Unique terms | Source |
|----------|-----------|-------|---------|--------------|--------|
| English  | 2,835     | 968   | 1,857   | 130          | OpenSubtitles, web |
| Hebrew   | 6,559     | 4,366 | 2,193   | 18           | HuggingFace corpora |
| Russian  | 1,259     | 538   | 721     | 15           | VKontakte (VK),Telegram |

---

## Reproducing the Results

### 1. Feature Extraction (Layer 20)
Open and run `layer20_features_extraction/Layer20_crosslinguistic_analysis.ipynb`.  
Requires: `transformer_lens`, `sae_lens`, access to Gemma-2-9B-IT and GemmaScope SAEs.

### 2. Permutation Test
```bash
python layer20_features_extraction/permutation_test.py \
    --json layer20_features_extraction/layer20_crosslinguistic_analysis.json \
    --n_perm 100000 \
    --plot
```

### 3. Generation Steering (Layer 9 & 20)
Run the notebooks in `layer9_steering_generation/` and `layer 20 cross linguistic steering generations/`.

### 4. Classification Steering
Run `layer20_steering_classification_/layer_20_classifcation+steering.ipynb`.

### 5. Vocabulary Projection
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

If you use this code or data, please cite:
```
@inproceedings{universalvibe2025,
  title     = {A Universal Vibe? Finding and Controlling Language-Agnostic Informal Register with SAEs},
  booktitle = {Proceedings of ACL},
  year      = {2025}
}
```
