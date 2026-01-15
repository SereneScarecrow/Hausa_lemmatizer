# Hausa NLP Toolkit
A simple Python toolkit for Hausa language text processing with POS tagging and lemmatization.

## What's Inside
Two main classes for Hausa text processing:

HausaPOSTagger - adds part-of-speech tags to Hausa text

HausaLemmatizer - finds base forms of words (lemmas)

## Model Attribution

This project uses the `masakhane/hausa-pos-tagger-afroxlmr` model from the Masakhane project for POS tagging. The model is based on AfroXLMR and was trained on Hausa text data.

## Dependencies

- transformers
- torch
- pandas