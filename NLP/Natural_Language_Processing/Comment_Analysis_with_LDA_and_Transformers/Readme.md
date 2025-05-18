# Topic Modeling of Survey Responses: Unsupervised Learning With LDA

## Author: Christian Lira Gonzalez  
## Summer 2023  (Revised on March 2025)


## Quick Overview

Introduction:
"This project is an application of unsupervised machine learning to analyze open-ended survey responses. The goal was to automate the process of extracting key themes from large-scale text data and enhance the insights through sentiment analysis and statistical validation."

Problem & Motivation:
"In many organizations, surveys generate a vast amount of textual responses, but analyzing them manually is time-consuming and prone to bias. My approach leverages Latent Dirichlet Allocation (LDA) to automatically categorize responses into meaningful topics. Additionally, I incorporated sentiment analysis to assess the emotional tone of each topic, helping decision-makers better interpret customer or employee feedback."

Methodology:
"The project involved several key steps:

1️⃣ Text Preprocessing: Removing stopwords, tokenizing, and lemmatizing the text.

2️⃣ Topic Modeling: Using LDA to discover hidden patterns and optimizing the number of topics based on coherence scores.

3️⃣ Visualization: Implementing T-SNE for topic clustering and word clouds to highlight sentiment-driven words.

4️⃣ Sentiment Analysis: Applying VADER to score each response’s sentiment and analyze sentiment variations across topics.

5️⃣ Statistical Validation: Running 1000 simulations to confirm the model’s robustness, followed by hypothesis testing to validate the significance of the results."


Key Findings:
"The results showed that the coherence scores remained stable across multiple simulations, meaning the topics generated were reliable. Additionally, the sentiment analysis revealed that responses were predominantly positive, providing valuable insights into customer or employee satisfaction. The hypothesis testing further confirmed that the model was statistically significant, ensuring that the extracted topics were not random."

Value & Business Impact:
*"This project can be applied in real-world business scenarios, such as:

✅ Customer Feedback Analysis – Identifying key pain points and trends in customer reviews.

✅ Employee Engagement Surveys – Understanding workplace sentiment and concerns.

✅ Market Research – Extracting insights from public sentiment on social media or product surveys.


By automating the text analysis process, businesses can make data-driven decisions more efficiently and focus on actionable insights instead of spending hours manually categorizing responses."*

Conclusion:
"This project demonstrates my ability to apply machine learning and natural language processing to real-world challenges. It showcases my skills in data preprocessing, topic modeling, sentiment analysis, and statistical validation—all of which are crucial for roles in data science, business intelligence, and analytics. I'm excited about the opportunity to bring this expertise to your team and solve complex data problems with innovative solutions."


---

## Project Structure
```
/Topic-Modeling-LDA/
│── data/                  # Folder for input survey dataset (CSV format)
│── notebooks/             # Jupyter Notebooks with code implementation
│── results/               # Output visualizations (word clouds, boxplots, histograms)
│── src/                   # Source code scripts for preprocessing, modeling & analysis
│── README.md              # Project documentation (this file)
│── requirements.txt       # Dependencies list for easy setup
```

---

## Installation & Setup
### Prerequisites
Ensure you have **Python 3.8+** installed. Install dependencies via:
```bash
pip install -r requirements.txt
```

### Required Libraries
```python
pandas
nltk
gensim
matplotlib
numpy
scikit-learn
seaborn
wordcloud
scipy
```
Install them using:
```bash
pip install pandas nltk gensim matplotlib numpy scikit-learn seaborn wordcloud scipy
```

---

## Usage

### 1. Data Preprocessing
- Cleans survey responses by **removing stopwords, tokenizing, and lemmatizing**.
- Handles missing values.
- Outputs a processed dataset ready for modeling.

Run:
```bash
python src/preprocess_data.py
```

### 2. Topic Modeling (LDA)
- Trains an **LDA model** with varying parameters.
- Identifies dominant **topics per document**.
- Evaluates **coherence scores** to optimize model performance.

Run:
```bash
python src/topic_modeling.py
```

### 3. Sentiment Analysis & Visualization
- Computes **sentiment scores** using **VADER**.
- Generates **word clouds** highlighting key sentiment-driven words.
- Creates **boxplots comparing sentiment scores per topic**.

Run:
```bash
python src/sentiment_analysis.py
```

### 4. Statistical Analysis & Model Validation
- Runs **1000 simulations** with different random configurations of LDA parameters (number of topics, passes, random seeds).
- Computes:
  - **Mean & standard deviation** of coherence and sentiment scores.
  - **95% confidence intervals** to assess stability.
  - **T-tests** for statistical validation.
- Produces **histograms of coherence & sentiment score distributions**.

Run:
```bash
python src/statistical_analysis.py
```

---

## Results & Insights
### Topic Coherence Scores
- The average coherence score was **0.306**, indicating **moderate topic coherence**.
- Confidence interval: **[0.264, 0.343]**, ensuring model stability.

### Sentiment Scores
- The mean sentiment score was **0.986**, suggesting that responses were **predominantly positive**.
- Confidence interval: **[0.780, 1.000]**, confirming consistency.

### Hypothesis Testing
- **Null Hypothesis:** The coherence score is equal to 0.3.
- **Result:** **p-value ≈ 8.33e-23** (much smaller than 0.05).
- **Conclusion:** Strong evidence against the null hypothesis; coherence is significantly different from 0.3.

---

## Error Handling & Debugging
- **Missing Values:** Handled before processing.
- **NameError (e.g., 'ldamodel' not defined):** Ensure correct LDA model reference.
- **Statistical Errors:** Ensure sufficient simulation runs for stable estimates.

---

## Future Enhancements
🚀 **Dynamic topic selection** based on coherence optimization.  
📊 **Interactive dashboards** with Streamlit for real-time analysis.  
🤖 **Deep learning-based topic modeling** (e.g., BERTopic, Transformer models).  

---


