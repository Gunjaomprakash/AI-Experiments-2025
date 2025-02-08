# Standard vs Contextual Retrieval Analysis

## Overview
This project evaluates the effectiveness of contextual retrieval compared to standard retrieval methods in RAG (Retrieval Augmented Generation) systems, inspired by [Anthropic's recent research](https://www.anthropic.com/research).

## Background
 Research paper (https://www.anthropic.com/news/contextual-retrieval) by Anthropic suggests that contextual retrieval methods may outperform traditional chunk-based retrieval. This experiment aims to validate these findings using real-world financial documents.

## Dataset
- Source: SEC 10-Q filings from [Docugami's KG-RAG Dataset](https://github.com/docugami/KG-RAG-datasets)
- Preprocessing: Documents chunked with metadata and summaries
- Format: Text chunks, summaries, and JSON metadata