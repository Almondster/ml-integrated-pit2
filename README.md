# AI Impact on Students: Major Category Predictor

An ML-powered web application that predicts a student's Major Category (Arts, Business, Humanities, Medical, or STEM) based on their AI usage patterns, study habits, and academic characteristics.

This project is developed as part of IT325 - Final Term - Performance Innovative Task 2.

---

## What This App Does

This application takes a trained machine learning classification model developed in Laboratory Activity 1 and integrates it into a highly polished, responsive Streamlit dashboard. It consists of two primary hubs:

### 1. Prediction Hub
The Prediction Hub serves as the user-facing inference interface where educators, researchers, or students can enter profile information to get real-time predictions.
* **Student Profile Form**: Accepts 13 distinct academic and behavioral inputs. The form features inline descriptions and validation ranges for each input.
* **On-the-Fly Input Validation**: Automatically validates inputs (such as validating that GPAs are between 1.0 and 4.0, and that study hours are non-negative) before running inference, displaying custom warning callouts if boundaries are violated.
* **Dynamic Prediction Cards**: Highlights the predicted Major Category in a styled hero container featuring unique icons and category-specific branding colors (Arts: Indigo, Business: Blue, Humanities: Green, Medical: Purple, STEM: Orange).
* **Interactive Probability Distribution**: Renders an interactive Plotly horizontal bar chart showing the model's confidence across all 5 possible major categories, sorted by probability.

### 2. Model Analysis & Performance Hub
This hub provides complete transparency regarding model construction, evaluation metrics, and dataset properties.
* **Top-Level Performance Cards**: Displays pre-calculated global model metrics: Accuracy (48.7%), Weighted F1-Score (45.9%), Macro F1-Score (39.8%), and number of target classes (5).
* **Per-Class Performance Table**: A clean tabular layout presenting category-specific Precision, Recall, F1-Score, and Support counts.
* **Interactive Confusion Matrix**: A custom Plotly heatmap showing predicted vs. actual classification intersections, equipped with custom hovering tooltips and themed color scales.
* **Dataset and Feature Specifications**: Outlines the original Kaggle dataset characteristics (50,000 rows, 80/20 train-test split) along with a detailed taxonomy of numerical and categorical variables.
* **Interactive Dataset Visualizations**: Pulls real student distribution charts if the dataset file is present, including:
  - Class Distribution Bar Chart: The volume of records across each Major Category.
  - GPA Box Plots: Pre-semester GPA ranges, quartiles, and outliers split by student majors.
  - GenAI Hours Violin Plots: Density curves and medians representing weekly AI usage across major categories.

---

## Extended Laboratory Activity

This application builds upon **Laboratory Activity 1**, which trained a multi-class classification pipeline using the scikit-learn framework.

### Machine Learning Pipeline Configuration
The model is saved as a complete serialized scikit-learn Pipeline object (`major_category_model_for_task2.pkl`), composed of:
1. **Preprocessor (`ColumnTransformer`)**:
   - **Numerical Pipeline**: Missing value imputation using `SimpleImputer` (median strategy) followed by normalization using `StandardScaler`.
   - **Categorical Pipeline**: Missing value imputation using `SimpleImputer` (most frequent strategy) followed by feature encoding using `OneHotEncoder` (handling unknown categories gracefully).
2. **Classifier (`LogisticRegression`)**:
   - Penalty: L2 Regularization
   - Solver: LBFGS
   - Maximum Iterations: 2000
   - Multi-class Strategy: Multinomial / One-vs-Rest

### Features Utilized for Inference

| Variable Name | Data Type | Preprocessing Method | Description |
|---|---|---|---|
| Year_of_Study | Categorical | OneHotEncoder | Current year level of the student |
| Pre_Semester_GPA | Numerical | StandardScaler | Cumulative GPA prior to the semester |
| Weekly_GenAI_Hours | Numerical | StandardScaler | Hours spent using generative AI per week |
| Tool_Diversity | Numerical | StandardScaler | Count of unique generative AI tools used |
| Primary_Use_Case | Categorical | OneHotEncoder | Primary objective for using AI tools |
| Prompt_Engineering_Skill | Categorical | OneHotEncoder | Self-rated prompt engineering competence |
| Traditional_Study_Hours | Numerical | StandardScaler | Weekly hours spent studying without AI |
| Perceived_AI_Dependency | Numerical | StandardScaler | Self-rated dependency rating (1 to 10) |
| Institutional_Policy | Categorical | OneHotEncoder | Academic institution's guidelines on AI |
| Anxiety_Level_During_Exams | Numerical | StandardScaler | Self-rated exam anxiety rating (1 to 10) |
| Post_Semester_GPA | Numerical | StandardScaler | Student's cumulative GPA after the semester |
| Skill_Retention_Score | Numerical | StandardScaler | Academic skill retention test score (10 to 100) |
| Burnout_Risk_Level | Categorical | OneHotEncoder | Assessed level of student academic burnout |

---

## Technical Stack

* **Frontend Dashboard**: Streamlit (featuring theme-adaptive custom CSS containers)
* **Machine Learning Runtime**: scikit-learn (v1.6.1), Joblib
* **Data Manipulation**: Pandas, NumPy
* **Interactive Data Visualization**: Plotly Express, Plotly Graph Objects

---

## Design System & User Interface Enhancements

* **Universal Theme Adaptability**: Custom stylesheets leverage theme-adaptive variables (`var(--text-color)`, `var(--background-color)`) and HSL-based CSS color mixing to dynamically adapt to both Streamlit's default Light and Dark modes.
* **Modern Sidebar Navigation**: Navigation buttons are styled to match modern product dashboards, featuring consistent text alignment, clean spacing, and custom inline SVG icons instead of default system lists.
* **Premium Typography**: Integrates Google Fonts (Inter and Plus Jakarta Sans) globally for enhanced legibility and layout appeal.
* **Resizer Cleanup**: Replaces raw, cluttered sidebar dividers with clean standard borders, preserving an interface free of unnecessary visual distractions.

---

## Project Structure

```
├── app.py                              # Main Streamlit application
├── ml-icon.svg                         # Vector favicon and branding assets
├── requirements.txt                    # Python dependencies
├── README.md                           # Documentation
├── model_metrics.json                  # Pre-computed model performance metrics
├── major_category_model_for_task2.pkl  # Serialized ML Pipeline object
├── data/
│   └── ai_student_impact_dataset (1).csv  # Original Kaggle student dataset
└── notebooks/
    └── laboratory-1.ipynb              # Lab Activity development notebook
```

---

## How to Run Locally

### Prerequisites
* Python 3.10 or higher
* pip package manager

### Execution Steps

1. **Verify Project Files**
   Ensure all files match the project layout above.

2. **Install Required Libraries**
   Run the following pip command to install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Dashboard**
   Execute the Streamlit application server:
   ```bash
   streamlit run app.py
   ```

4. **Access the Web Interface**
   Open your preferred browser and navigate to the local network port (typically `http://localhost:8501`).
