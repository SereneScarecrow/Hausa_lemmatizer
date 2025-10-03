# Hausa Lemmatizer

A rule-based lemmatizer for the Hausa language that combines POS tagging with morphological analysis.

## Features

- POS tagging using pre-trained Hausa model
- Rule-based lemmatization with dictionary fallback
- Multiple output formats: word_POS_lemma, lemma-only, combined tokens
- Handles plural nouns, verbs, pronouns, and numerals

## Model Attribution

This project uses the `masakhane/hausa-pos-tagger-afroxlmr` model from the Masakhane project for POS tagging. The model is based on AfroXLMR and was trained on Hausa text data.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from hausa_lemmatizer import HausaLemmatizer

lemmatizer = HausaLemmatizer()
text = "Shin matsalar dabanci ta gari hukumomi ne a Kano?"

# Get JSON analysis
analysis = lemmatizer.analyze_sentence(text)

# Get formatted output
formatted = lemmatizer.get_lemma_sentence(text, "underscore")
```

## Output Formats

- underscore: word_POS_lemma
- lemma_only: space-separated lemmas  
- combined: merged tokens with POS tags

## Dependencies

- transformers
- torch
