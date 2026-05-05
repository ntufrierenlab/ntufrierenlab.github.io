# Paper Knowledge Base — Design Document

## Overview
A Hugo-based static site hosted on GitHub Pages (ntufrierenlab.github.io) that serves as a lab paper knowledge base. Papers are fetched from arXiv, summarized by Claude, and presented in structured markdown.

## Users
NTU Frieren Lab team members.

## Two Core Features

### 1. Daily Auto-Fetch
- **Trigger**: GitHub Actions cron (daily)
- **Source**: arXiv API, filtered by categories (cs.CV, eess.IV) + keywords
- **Initial keywords**: "color constancy", "auto white balance"
- **Limit**: 5 papers/day
- **Process**: Fetch → Download PDF → Claude summarize → Commit MD → Hugo deploy

### 2. Manual Upload
- **Trigger**: CLI command `./scripts/add-paper.sh <arXiv-URL>`
- **Process**: Download PDF → Claude summarize → Generate MD → Commit

## Topic Management
- Keywords stored in `config/topics.yaml`
- Website has an "Add Topic" button that creates a GitHub Issue or PR
- Topics are configurable: name, arXiv categories, keywords

## Paper Summary Structure (paper-reading skill output)
Each paper summary MD includes:
- Title, authors, arXiv link, date
- One-line summary
- Key contributions (bulleted)
- Core insights
- Key data / experiment results
- Strengths
- Weaknesses
- Potential improvements
- Related work connections
- Tags / keywords

## Visual Design
- **Style**: Clean academic knowledge base, generous whitespace
- **Colors**: Deep blue/teal accent, warm white background, dark mode support
- **Typography**: Inter (sans-serif) for UI, optimized for readability
- **Layout**:
  - Homepage: card grid of papers, filterable by topic/date
  - Paper page: structured sections with clear hierarchy
  - Sidebar: topic navigation + search
  - "Add Topic" button in navigation

## Tech Stack
- Hugo (static site generator)
- Custom theme (built from scratch for this project)
- GitHub Actions (daily cron + deploy)
- Python script (arXiv API fetching)
- Claude CLI / API (summarization)
- GitHub Pages (hosting)

## Repository Structure
```
ntufrierenlab.github.io/
├── config/
│   └── topics.yaml          # tracked keywords & categories
├── content/
│   └── papers/
│       └── 2026-02-06-paper-title.md
├── themes/
│   └── knowbase/            # custom Hugo theme
├── scripts/
│   ├── daily-fetch.py       # arXiv auto-fetch
│   ├── add-paper.sh         # manual paper add
│   └── summarize.sh         # Claude summarization
├── .github/
│   └── workflows/
│       ├── daily-papers.yml  # cron job
│       └── deploy.yml        # Hugo build & deploy
└── hugo.toml
```
