# Oculomotor Movement Analysis

A Python-based project for the analysis of **eye-tracking and oculomotor data**, with a particular focus on **fixations, saccades and regressions during reading**.

The project investigates differences in oculomotor behaviour under different font-size conditions and compares a normative group with presbyopic participants.

> **Note:** This repository is part of ongoing research. Only a high-level overview of the results is reported here.

## 👁️ Project Overview

Eye-tracking provides quantitative information about visual behaviour and can be used to investigate how reading conditions affect oculomotor activity.

This project analyses three main measures:

* Fixations
* Saccades
* Regressions

The analysis evaluates how these measures change across different font sizes and how individual participants compare with a normative reference distribution.

## 🎯 Objectives

The main objectives of the project are:

* Analyse oculomotor behaviour across different font sizes
* Compare normative and presbyopic participants
* Define normative references using quartile-based thresholds
* Evaluate the stability of the normative reference sample
* Classify individual participants relative to the normative distribution
* Assess the robustness of the classifications
* Identify individual oculomotor profiles showing consistently elevated activity

## 🔬 Analysis Pipeline

The analysis pipeline includes:

1. Data preprocessing and cleaning
2. Descriptive analysis of oculomotor measures
3. Comparison between normative and presbyopic groups
4. Statistical inference
5. Definition of normative quartile thresholds
6. Leave-one-out stability analysis
7. Classification of presbyopic participants
8. Robustness analysis of the classifications
9. Evaluation of fixation, saccade and regression concordance

## 📊 Statistical Analysis

Different statistical and computational approaches are used throughout the project, including:

* Descriptive statistics
* Distribution analysis
* Mann–Whitney U tests
* Multiple-comparison correction
* Quartile-based classification
* Leave-one-out validation
* Correlation and concordance analysis

The use of leave-one-out procedures allows the stability of the normative thresholds to be evaluated without classifying a subject using thresholds influenced by that same subject.

## 📈 Data Visualization

Several visual representations are generated to investigate both group-level and individual behaviour, including:

* Mean trends across font sizes
* Variability and error-bar plots
* Percentage differences
* Quartile reference curves
* Participant classification heatmaps
* Stability heatmaps

These visualizations support the interpretation of both overall trends and individual variability.

## 🔎 Preliminary Findings

The analyses reveal several interesting patterns.

At a descriptive level, **oculomotor activity generally increases as font size decreases**, with presbyopic participants tending to show higher values, particularly for fixations and saccades.

The differences between groups appear more evident under more demanding reading conditions, while regressions show a more heterogeneous pattern.

The normative quartile-based reference system also showed **good overall stability**, allowing individual participants to be compared with the normative distribution.

An important aspect emerging from the analysis is the presence of substantial **inter-individual variability**. Rather than suggesting a uniform shift affecting all presbyopic participants, the results indicate that some individuals present consistently elevated oculomotor profiles compared with normative references.

Further quantitative results, detailed statistical outcomes and the complete interpretation of these findings are intentionally not reported in this repository because they are part of an ongoing scientific publication.

## 🛠️ Technologies

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **SciPy**
* **Statsmodels**
* **JSON**

## 📁 Repository Structure

```text
Analisi-dei-movimenti-oculomotori/
│
├── Analisi Statistiche Finali.py
├── Prova_analisi_dati_di_gaze.py
└── .gitignore
```

### `Analisi Statistiche Finali.py`

Contains the main analysis pipeline, including:

* Dataset preprocessing
* Descriptive statistics
* Normative vs. presbyopic comparisons
* Statistical testing
* Quartile definition
* Participant classification
* Robustness analyses
* Visualization of results

### `Prova_analisi_dati_di_gaze.py`

Contains preliminary processing and exploratory analysis of the eye-tracking data.

## 🚀 Running the Project

Clone the repository:

```bash
git clone https://github.com/thomas930821/Analisi-dei-movimenti-oculomotori.git
```

Enter the project directory:

```bash
cd Analisi-dei-movimenti-oculomotori
```

Install the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels
```

Then run the analysis script:

```bash
python "Analisi Statistiche Finali.py"
```

## ⚠️ Data Availability

The experimental dataset used for the research is **not included in this repository**.

Some local paths in the scripts may need to be adapted before execution.

The repository primarily provides the analysis code and methodological workflow developed for the research project.

## 🧠 Skills Demonstrated

This project demonstrates experience with:

* Scientific data analysis
* Eye-tracking data processing
* Statistical hypothesis testing
* Experimental data analysis
* Data visualization
* Validation techniques
* Quartile-based classification
* Robustness analysis
* Python programming
* Reproducible analytical workflows

## 📄 Research Status

The complete results and interpretation of this study are currently being prepared for **scientific publication**.

For this reason, the repository intentionally contains only a limited summary of the findings.

## 👤 Author

**Thomas Farinelli**

Computer Science

GitHub: [thomas930821](https://github.com/thomas930821)
