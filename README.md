# **Drug Review Sentiment Analysis Using NLP**

This project analyzes the **Drug Review Dataset** to uncover insights into drug usage, effectiveness, and patient sentiment. It employs **Natural Language Processing (NLP)** techniques and a general **sentiment analysis model** from Hugging Face Transformers.

---

## **Table of Contents**

1. [Introduction](#introduction)
2. [Dataset Overview](#dataset-overview)
3. [Project Objectives](#project-objectives)
4. [Technologies Used](#technologies-used)
5. [Analysis and Insights](#analysis-and-insights)
6. [Sentiment Analysis Approach](#sentiment-analysis-approach)
7. [Results and Findings](#results-and-findings)
8. [Future Scope](#future-scope)
9. [How to Run](#how-to-run)

---

## **Introduction**

Understanding patient reviews is crucial for evaluating the effectiveness of drugs and improving healthcare outcomes. This project uses NLP and transformer models to analyze user reviews, perform sentiment classification, and draw actionable insights.

---

## **Dataset Overview**

* **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Drug+Review+Dataset+%28Drugs.com%29)
* **Shape**: 161,297 rows and 7 columns
* **Features**:

  * `drugName`: Name of the drug.
  * `condition`: Medical condition treated by the drug.
  * `review`: Patient's textual review.
  * `rating`: 10-star rating reflecting satisfaction.
  * `date`: Date of the review.
  * `usefulCount`: Number of users who found the review helpful.

---

## **Project Objectives**

1. **Exploratory Data Analysis (EDA)**:

   * Explore trends in drug usage and patient feedback.
   * Understand rating and condition distributions.
2. **Sentiment Analysis**:

   * Classify reviews into positive, negative, or neutral categories.
   * Correlate sentiment with ratings and user helpfulness.
3. **Transformer-Based Sentiment Analysis**:

   * Leverage a general sentiment analysis model for effective classification.

---

## **Technologies Used**

* **Programming Language**: Python
* **Libraries**:

  * Data Manipulation: Pandas, NumPy
  * Visualization: Matplotlib, Seaborn
  * NLP: Hugging Face Transformers
  * Model Evaluation: Scikit-learn
* **Model**: Pre-trained general sentiment analysis model from Hugging Face.

---

## **Analysis and Insights**

1. **EDA Results**:

   * **Popular Drugs**: Levonorgestrel, Gabapentin, and Bupropion.
   * **Common Conditions**: Birth Control, Depression, Pain, and Anxiety.
   * **Ratings**: Polarized distribution, with most reviews rated as 1 or 10.
2. **Sentiment Distribution**:

   * Positive: 62.6%
   * Negative: 33%
   * Neutral: 4.3%
3. **Useful Feedback**:

   * Drugs with higher useful counts often had detailed reviews.

---

## **Sentiment Analysis Approach**

1. **Preprocessing**:

   * Tokenized the dataset for input to the sentiment analysis model.
2. **Transformer Model**:

   * Used a general Hugging Face sentiment analysis pipeline.
   * Predicts labels as positive, negative, or neutral.

---

## **Results and Findings**

* Positive sentiment correlates with higher ratings (6-10).
* Negative sentiment correlates with lower ratings (1-4).
* August and October had the highest review counts, reflecting seasonal trends in drug usage.

---

## **Future Scope**

* **Model Improvements**:

  * Train a custom transformer model using domain-specific data.
  * Add multi-class classification for side-effect and satisfaction ratings.
* **Applications**:

  * Develop a web application for real-time analysis of drug reviews.

---

## **How to Run**

1. Clone this repository:

   ```bash
   git clone https://github.com/your_username/drug-review-sentiment-analysis.git
   ```
2. Install required libraries:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook:

   ```bash
   jupyter notebook Drugs_Review_Analysis.ipynb
   ```
4. For sentiment analysis using Hugging Face:

   ```bash
   pip install transformers datasets
   ```

---

## **Acknowledgments**

* **Dataset**: UCI Machine Learning Repository.
* **Model**: Hugging Face Sentiment Analysis Pipeline.

Feel free to contribute by raising issues or submitting pull requests!

---
