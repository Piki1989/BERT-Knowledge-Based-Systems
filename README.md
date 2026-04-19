# BERT-Knowledge-Based-Systems
Official repository for paper "Selecting a subset of large language models ensemble for subject-specific text embedding tasks using fuzzy sets approach" T.Hachaj, M.Pikus, T.Frąś, J.Wąs

# LLM Ensemble Embedding Optimization

This repository contains a pipeline for building and optimizing domain-specific text embeddings using ensembles of large language models (LLMs). The system combines domain-adaptive fine-tuning, contrastive embedding learning, and heuristic ensemble selection to improve query–document matching performance.

## What this project does

The goal is to improve semantic retrieval in specialized scientific domains by combining multiple embedding models into an optimized ensemble. Instead of relying on a single model, we evaluate a set of models and automatically select the best subset that maximizes retrieval accuracy.

The approach treats query–document matching as a measurable decision problem and uses a fuzzy-based scoring function to evaluate ensemble performance. A genetic algorithm is then used to search for an optimal combination of models.

## Pipeline overview

The system is built in three main stages:

1. **Data processing**
   - Collection and parsing of scientific papers (HTML format)
   - Extraction of abstracts and main text
   - Cleaning and segmentation into training chunks

2. **Embedding model training**
   - Domain-adaptive pretraining using masked language modeling
   - Sentence-level contrastive training using Sentence Transformers
   - Optimization of embeddings for semantic similarity tasks

3. **Ensemble optimization**
   - Evaluation of multiple embedding models on query–document pairs
   - Definition of a unified correctness score for retrieval performance
   - Search for optimal model subsets using a genetic algorithm
   - Selection of best-performing ensemble configuration

## Key idea

Instead of using a single embedding model, we evaluate a pool of models (fine-tuned and non-fine-tuned) and automatically construct an ensemble that maximizes retrieval accuracy on a given corpus. Each model votes on query–document matches, and the system selects the combination that produces the most consistent correct mappings.

## Optimization

The ensemble selection problem is treated as a combinatorial search task. A genetic algorithm is used to explore possible subsets of models. Each candidate subset is evaluated based on its retrieval performance, and the best-performing configuration is selected.

## Use case

The framework is designed for:
- domain-specific search engines
- scientific literature retrieval
- embedding model benchmarking
- ensemble-based NLP systems

## Summary

This project provides a full pipeline for:
- training domain-specific embeddings
- evaluating multiple LLM-based encoders
- automatically building optimized embedding ensembles

The result is a scalable and reproducible approach to improving semantic search in specialized datasets.

papers_info.txt -> This file contains the authors, titles, and DOIs of all publications that were used in the dataset.
