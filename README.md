# Vietnam National Exam — English Score & Socioeconomic Analysis

# Quickstart [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://phnam05-english-score-analysis.streamlit.app/)

## Overview

This project explores how socioeconomic conditions affect English language performance across Vietnam's 63 provinces, using data from the 2024 National university entrance exam (THPT Quốc Gia). Exam score data is merged with provincial statistics on average income, household computer access, and poverty rate to uncover structural disparities in English education nationwide.

The project has two layers:

- **An interactive analytics dashboard** — four tabs covering score distributions, province-level breakdowns, regional comparisons, and socioeconomic correlations
- **An AI agent** — a natural language interface that lets users ask questions about the data and get plain-English answers powered by Gemini AI

---

## Key Findings

- Provinces with higher income and computer access rates score significantly higher in English
- Poverty rate shows a strong negative correlation with English exam scores
- Clear regional gaps exist between North, Central, and South Vietnam
- Some provinces with above-average income still underperform — suggesting income alone doesn't explain the gap

---

## AI Agent

The app includes a tool-calling AI agent built on top of the existing analysis.

**How it works:**

1. You ask a question in plain English
2. Gemini reads the question and decides which analysis tool(s) to run
3. Python functions execute the actual data analysis using pandas
4. Gemini interprets the results and returns a clear, plain-English answer

The LLM never touches raw data directly — it only sees pre-computed summaries from modular analysis tools, keeping responses fast and grounded in actual numbers.

**Available tools:**

- Top / bottom provinces by score
- Regional summary (North, Central, South)
- Correlation between any two variables
- Outlier detection (high income but low scores)
- Province deep-dive and side-by-side comparison
- National KPI summary

**Supported models (free tier):**

- Gemini 2.5 Flash *(recommended)*
- Gemini 2.5 Flash Lite
- Gemma 3 27B

To use the agent, get a free API key at [aistudio.google.com](https://aistudio.google.com) and paste it into the app.

---

## Growth Analytics Parallels

This project uses the same analytical patterns found in product growth and marketing analytics:

| What this project does | Growth analytics equivalent |
|---|---|
| Outlier detection — high income but low scores | Identifying churned users with high engagement potential |
| RFM-style province segmentation | User segment prioritization for campaigns |
| Score funnel — registered → passed → high scorer | Acquisition → activation → retention funnel |
| Correlation of PC access to scores | Feature impact analysis on conversion rate |
| Regional performance gaps | Cohort comparison across user segments |

---

## Datasets

- 2024 THPT national exam scores (English subject, ~1M student records)
- Provincial average monthly income — GSO Statistical Yearbook 2023
- Household computer access rate by province — GSO 2023
- Multidimensional poverty rate by province — GSO 2023

---

## Tools & Methods

- **Python** — Pandas, NumPy, Matplotlib, Seaborn, Altair, Streamlit
- **AI** — Gemini API, tool-calling agent architecture
- Exploratory Data Analysis (EDA)
- Linear regression & correlation analysis
- Outlier detection and provincial segmentation

---

## Visualizations

View the full visual report here: [Canva Report](https://canva.link/0wav7jz9wez7anm)

---

## Run it locally

**1. Clone the repository**
```bash
git clone https://github.com/phnam05/vietnam-english-score-analysis.git
cd vietnam-english-score-analysis
```

**2. Set up a virtual environment**
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the app**
```bash
streamlit run english-app.py
```

Then open the **🤖 AI Agent** tab, paste your free Gemini API key, and start asking questions.
