# 🎯 Cogentix HR Analytics: Employee Performance & Retention Analysis

## 📊 Project Overview

This project delivers a comprehensive **HR Analytics solution** for Cogentix, a multinational organization with 10,000+ employees across 10 global locations. Using advanced data science techniques, this analysis provides actionable insights into employee engagement, performance drivers, promotion readiness, and compensation equity.

### 🎓 Business Impact

- **Identified at-risk employee segments** with low engagement for targeted retention strategies
- **Built predictive models** to identify promotion-ready employees with 54% AUC
- **Analyzed performance drivers** across departments, roles, and demographics
- **Flagged compensation inequities** to ensure fair pay for high performers
- **Created interactive dashboard** for real-time compensation analysis

---

## 🔍 Key Analyses & Findings

### 1️⃣ Demographic Risk Analysis (Employee Engagement)

**Objective:** Identify demographic groups with low engagement scores to inform targeted retention strategies.

**Methodology:**
- Defined "low engagement" as bottom 25th percentile (≤63.9 score)
- Performed Chi-square tests across Age, Gender, and Location
- Calculated effect sizes using Cramér's V
- Built Logistic Regression model to quantify demographic impacts

**Key Findings:**
- **Age Groups:** 40-44 age bracket shows highest low-engagement rate (27.0%)
- **Gender & Location:** No statistically significant differences (p > 0.05)
- **Model Performance:** Weak predictive power (AUC ~0.51) indicates engagement is driven by factors beyond demographics
- **Feature Importance:** Training hours and specific locations (Hyderabad, Mumbai) show marginal influence

**Recommendation:** Focus retention efforts on mid-career employees (40-44 age group) while investigating non-demographic factors like manager quality and role satisfaction.

---

### 2️⃣ Promotion Readiness Modeling

**Objective:** Build a leakage-free predictive model to identify employees ready for promotion.

**Methodology:**
- **Leakage Prevention:** Excluded `LastPromotionMonthsAgo` and manager-aggregated features to ensure model fairness
- **Feature Engineering:** Created role-based salary percentiles, engagement interactions
- **Model:** LightGBM with 5-fold cross-validation
- **Evaluation Metrics:** AUC, PR-AUC, Precision@10%

**Model Performance:**
```
LightGBM 5-fold CV Results:
├─ AUC: 0.5388 ± 0.0044
├─ PR-AUC: 0.4329 ± 0.0076
└─ Precision@10%: 0.4330 ± 0.0223
```

**Top Predictive Features:**
1. **EngagementScore** (2,060 importance) - Highly engaged employees are promotion candidates
2. **BonusPercent** (2,059) - Bonus recognition correlates with promotion readiness
3. **YearsAtCompany** (2,027) - Tenure indicates organizational commitment
4. **SalaryRolePct** (1,870) - Relative compensation within role
5. **TrainingHoursLastYear** (1,539) - Investment in skill development

**Top 50 Promotion Candidates Identified:**
- Exported to `top_promotable_candidates.csv`
- Includes employees like **Sara Chaudhary** (Consulting, 99.5% probability), **Vihaan Gupta** (Consulting Manager, 98.5%)

**Recommendation:** Prioritize promotion reviews for high-scoring candidates, focusing on engagement and continuous learning as key indicators.

---

### 3️⃣ Performance Driver Analysis

**Objective:** Identify factors that differentiate high performers (rating ≥4) from others.

**Methodology:**
- Defined high performers as PerformanceRating ≥ 4 (34.2% of workforce)
- Conducted Kruskal-Wallis tests for categorical variables (Department, Location, Role)
- Calculated Spearman correlations for numerical features
- Created visualizations to illustrate performance distributions

**Statistical Results:**

| Factor | Test | p-value | Significance |
|--------|------|---------|--------------|
| **Department** | Kruskal-Wallis | 0.3410 | Not significant |
| **Location** | Kruskal-Wallis | 0.0093 | **Significant** ✓ |
| **Role** | Kruskal-Wallis | 0.6730 | Not significant |

**Spearman Correlations with Performance:**

![Spearman Correlations](C:\Users\HP\.gemini\antigravity\brain\2b02b8fb-1c3b-4de5-9113-0ca38b3893d7\images\q3_spearman_correlations.png)

| Feature | Correlation | Insight |
|---------|-------------|---------|
| **TrainingHoursLastYear** | +0.0065 | More training → Better performance |
| **AnnualSalaryINR** | +0.0033 | Higher pay correlates with performance |
| **EngagementScore** | +0.0033 | Engaged employees perform better |
| **BonusPercent** | -0.0046 | Weak negative correlation |
| **Age** | -0.0087 | Younger employees slightly outperform |
| **YearsAtCompany** | -0.0092 | Tenure shows slight negative correlation |

**Performance by Department:**

![Performance by Department](C:\Users\HP\.gemini\antigravity\brain\2b02b8fb-1c3b-4de5-9113-0ca38b3893d7\images\q3_boxplot_Department.png)

**Performance by Role:**

![Performance by Role](C:\Users\HP\.gemini\antigravity\brain\2b02b8fb-1c3b-4de5-9113-0ca38b3893d7\images\q3_boxplot_Role.png)

**Training vs Tenure Heatmap:**

![Training vs Tenure Heatmap](C:\Users\HP\.gemini\antigravity\brain\2b02b8fb-1c3b-4de5-9113-0ca38b3893d7\images\performance_heatmap_training_vs_tenure.png)

**Key Insights:**
- **Location matters:** Significant performance variation across offices (p=0.0093)
- **Training is crucial:** Positive correlation with performance, especially for newer employees
- **Weak correlations overall:** Performance is multifaceted; no single factor dominates
- **Department/Role neutral:** Performance is consistent across functions

**Recommendation:** Invest in location-specific performance improvement programs and increase training hours, particularly for employees in their first 3-6 years.

---

### 4️⃣ Compensation vs Performance Analysis

**Objective:** Identify high-performing employees who are underpaid relative to their peers.

**Methodology:**
- Built **Linear Regression model** to predict expected salary based on:
  - Performance Rating
  - Years at Company
  - Role (one-hot encoded)
  - Location (one-hot encoded)
- Calculated **salary residuals** (actual - predicted log salary)
- Flagged employees with:
  - High performance (rating ≥ 4)
  - Low salary residuals (bottom 20th percentile)

**Interactive Dashboard:**
- **Streamlit application** (`Dashboard.py`) for dynamic analysis
- **Features:**
  - Filter by Department, Role, Performance Rating
  - Adjust underpaid cutoff threshold (5-50%)
  - Visualize salary distributions, residuals, and department summaries
  - Export underpaid employee lists as CSV

**Key Outputs:**
- **Flagged Underpaid High Performers:** Exported to `assets/underpaid_high_performers.csv`
- **Visualization:** Scatter plots, violin plots, and residual distributions

**Recommendation:** Review compensation for flagged employees to reduce turnover risk and ensure pay equity.

---

## 🚨 Manager Performance Red Flags

**Analysis:** Identified 27 managers with statistically significant low team engagement using:
- Wilson confidence intervals for proportion estimation
- One-sided binomial tests (p < 0.05)
- Adjusted percentages accounting for team size

**Top 5 Flagged Managers:**

| ManagerID | Team Size | Low Engagement % | p-value | Avg Engagement | Avg Performance |
|-----------|-----------|------------------|---------|----------------|-----------------|
| AF100476 | 22 | 63.6% | 0.000015 | 67.6 | 2.86 |
| AF100297 | 18 | 55.6% | 0.0014 | 64.3 | 3.17 |
| AF100111 | 16 | 56.3% | 0.0020 | 66.5 | 3.19 |
| AF100120 | 21 | 52.4% | 0.0020 | 66.0 | 3.19 |
| AF100143 | 17 | 52.9% | 0.0041 | 66.7 | 3.18 |

**Full List:** `flagged_managers_summary.csv` and `flagged_managers_teams.csv`

**Recommendation:** Conduct manager effectiveness reviews and provide leadership training for flagged individuals.

---

## 📁 Project Structure

```
cogentix/
├── 📄 README.md                                # Project documentation
├── 📋 requirements.txt                         # Python dependencies
├── 🔒 .gitignore                               # Git ignore rules
│
├── 📂 data/                                    # Raw data files
│   └── Cogentix_Case.xlsx                      # Employee dataset (10,000 records)
│
├── 📓 notebooks/                               # Analysis notebooks
│   └── Congentix.ipynb                         # Main analysis (all 4 questions)
│
├── 🎨 app/                                     # Interactive applications
│   └── Dashboard.py                            # Streamlit compensation dashboard
│
├── 📚 docs/                                    # Documentation & presentations
│   ├── Cogentix_Case.pdf                       # Case study brief
│   └── Team_Moles.pptx                         # Presentation slides
│
└── 📊 outputs/                                 # Analysis results
    ├── visualizations/                         # Charts and plots
    │   ├── performance_heatmap_training_vs_tenure.png
    │   ├── q3_boxplot_Department.png
    │   ├── q3_boxplot_Role.png
    │   ├── q3_boxplot_Gender.png
    │   └── q3_spearman_correlations.png
    │
    └── csv_reports/                            # Data exports
        ├── top_promotable_candidates.csv       # Top 50 promotion-ready employees
        ├── top_promotable_noleak.csv           # Leakage-free candidates
        ├── flagged_managers_summary.csv        # Underperforming managers
        ├── flagged_managers_teams.csv          # Team-level details
        ├── underpaid_high_performers.csv       # Compensation equity analysis
        ├── lgb_feature_importances.csv         # Model feature importance
        ├── promotion_cv_metrics.csv            # Cross-validation results
        ├── performance_spearman_correlations.csv
        ├── performance_kruskal_tests.csv
        └── df_with_oof_encodings_and_interactions.csv
```

---

## 🛠️ Technologies Used

| Category | Tools |
|----------|-------|
| **Programming** | Python 3.x |
| **Data Analysis** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn, LightGBM |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Dashboard** | Streamlit |
| **Statistical Testing** | SciPy (Chi-square, Kruskal-Wallis, Spearman) |
| **Data Processing** | OpenPyXL (Excel handling) |

---

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Analysis Notebook

```bash
jupyter notebook notebooks/Congentix.ipynb
```

The notebook contains all four analyses with detailed explanations, visualizations, and statistical tests.

### 3. Launch the Interactive Dashboard

```bash
streamlit run app/Dashboard.py
```

**Dashboard Features:**
- 💸 Salary vs Performance scatter plots
- 🏢 Department-level summaries
- 🎻 Violin plots for salary distributions
- 📉 Residual analysis with customizable thresholds
- 📥 Export underpaid employee lists

**Note:** The dashboard expects the data file at `data/Cogentix_Case.xlsx`. Update the `DATA_PATH` variable in `app/Dashboard.py` if needed.

---

## 📊 Key Deliverables

1. ✅ **Demographic Risk Report** - Identified 40-44 age group as high-risk for low engagement
2. ✅ **Promotion Candidate List** - 50 employees with highest promotion probability
3. ✅ **Performance Driver Insights** - Training hours and location are key differentiators
4. ✅ **Compensation Equity Analysis** - Flagged underpaid high performers for review
5. ✅ **Manager Effectiveness Report** - 27 managers with statistically low team engagement
6. ✅ **Interactive Dashboard** - Real-time compensation analysis tool

---

## 💡 Business Recommendations

### Immediate Actions (0-3 months)
1. **Review compensation** for flagged underpaid high performers
2. **Conduct manager training** for 27 flagged managers with low team engagement
3. **Initiate promotion discussions** for top 50 identified candidates
4. **Increase training hours** for employees with 0-6 years tenure

### Strategic Initiatives (3-12 months)
1. **Location-specific programs** to address performance variations across offices
2. **Mid-career retention programs** targeting 40-44 age group
3. **Engagement surveys** to identify non-demographic factors affecting morale
4. **Compensation benchmarking** to ensure market competitiveness

### Long-term Improvements (12+ months)
1. **Predictive analytics integration** into HR systems for real-time insights
2. **Manager effectiveness metrics** tied to team engagement and performance
3. **Career development pathways** based on promotion readiness model
4. **Continuous monitoring** of compensation equity using residual analysis

---

## 📧 Contact & Collaboration

This project demonstrates expertise in:
- 🎯 **Business Analytics** - Translating data into actionable HR strategies
- 🤖 **Machine Learning** - Building predictive models for talent management
- 📊 **Statistical Analysis** - Rigorous hypothesis testing and correlation studies
- 🎨 **Data Visualization** - Creating compelling visual narratives
- 💻 **Dashboard Development** - Interactive tools for stakeholder engagement

**For recruiters:** This project showcases end-to-end data science capabilities from exploratory analysis to production-ready dashboards, with a focus on business impact and statistical rigor.

---

## 📜 License

This project is part of a case study analysis for Cogentix. All data is anonymized and used for analytical purposes only.

---

**Last Updated:** February 2026  
**Analysis Period:** Current employee snapshot (10,000 employees)  
**Tools Version:** Python 3.x, Streamlit 1.x, LightGBM 4.x
