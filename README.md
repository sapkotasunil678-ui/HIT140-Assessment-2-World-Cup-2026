# HIT140 Assessment 2 — FIFA World Cup 2026 Data Analysis

## Overview
This repository contains our group's Objective 1 submission for HIT140 Assessment 2: four distinct analytic tasks investigating team and player performance at the FIFA World Cup 2026, using real match and player statistics.

Each task follows the same six-step data science pipeline:
**Question → Data Wrangling → Sampling → Descriptive Statistics → Confidence Interval → Hypothesis Test (t-test)**

All data wrangling and analysis were carried out in Python (pandas, scipy, matplotlib). Excel was used only for preparing raw datasets, per the assessment brief.

## Data Sources
- [FIFA Official Website](https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/statistics)
- [The Stats Don't Lie](https://www.thestatsdontlie.com/football/world-cup-2026/)
- [FBref](https://fbref.com/en/)

## Group Members & Tasks

| Task | Analytic Question | Owner | Test Type |
|------|-------------------|-------|-----------|
| 1 | Is there a significant difference in average fouls per 90 minutes between midfielders and defenders? | Md Hasibul Raihan | Two-sample t-test |
| 2 | On average, was the age of players who started at least one match significantly different from the widely cited peak performance age of 27? | Abishek Rajeshkumar | One-sample t-test |
| 3 | Is there a significant difference in average goalkeeper saves per 90 minutes between teams that reached the knockout stage and teams eliminated in the group stage? | Sunil Sapkota | Two-sample t-test |
| 4 | Do teams that advanced past the group stage have significantly higher average possession than teams eliminated in the group stage? | Rejan Sapkota | Two-sample t-test |

