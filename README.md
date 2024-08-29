**Financial Sentiment Analysis Using Hybrid Approach**

**Overview**

This project focuses on Financial Sentiment Analysis (FSA) using a
hybrid approach that combines both machine learning (ML) and deep
learning (DL) models. The research aims to evaluate and compare the
performance of various individual models---such as Categorical Naive
Bayes (CNB), Random Forest (RF), LightGBM, K-Nearest Neighbors (KNN),
and Multi-Layer Perceptron (MLP)---and their effectiveness when
integrated into hybrid models through ensemble voting. The goal is to
determine if a hybrid approach can outperform individual models in
classifying the sentiment of financial text data.

**Objectives**

-   **Review Existing Research**: Conduct a comprehensive literature
    review on financial sentiment analysis, focusing on the use of ML
    and DL models for sentiment classification.

-   **Data Collection and Preprocessing**: Collect and preprocess three
    financial sentiment analysis datasets, ensuring the removal of
    irrelevant information, duplicates, and null values.

-   **Text Normalization and Feature Engineering**: Apply text
    normalization techniques like stemming, lemmatization, and feature
    engineering to extract relevant sentiment indicators.

-   **Data Splitting**: Split the data into training, validation, and
    testing sets.

-   **Model Implementation**: Implement individual ML and DL models
    (CNB, RF, LightGBM, KNN, and MLP) and optimize them using
    hyperparameter tuning.

-   **Ensemble Modeling**: Create hybrid models using ensemble voting by
    combining predictions from individual models.

-   **Model Evaluation**: Evaluate and compare the performance of
    individual and ensemble models using metrics such as accuracy,
    precision, recall, and F1-score.

-   **Determine Best Model**: Identify the best-performing model or
    combination of models for financial sentiment analysis.

**Features**

-   **Data Preprocessing**: Includes removal of duplicates, handling of
    null values, label encoding, and NLP preprocessing.

-   **Vectorization**: Converts text data into numerical features using
    TF-IDF vectorization.

-   **Data Balancing**: Balances the dataset using the Random Over
    Sampler technique.

-   **Model Implementation**: Implements various individual and combined
    ML/DL models.

-   **Hyperparameter Tuning**: Optimizes model performance using Grid
    Search.

-   **Ensemble Modeling**: Combines predictions from multiple models
    using the voting classifier.

-   **Performance Evaluation**: Provides detailed performance metrics
    and visualizations for model evaluation.

**Installation**

1.  **Clone the repository**:

> bash
>
> Copy code
>
> git clone
> https://github.com/Charan91120/Data_science_final_project
>
> cd financial-sentiment-analysis-hybrid

2.  **Install the required packages**:

> bash
>
> Copy code
>
> pip install -r requirements.txt

3.  **Download the dataset**:

    -   Download the Financial Sentiment Analysis dataset from
        [[Kaggle]{.underline}](https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis).

    -   Place the dataset in the data/ directory.

**Usage**

1.  **Data Preprocessing**:

    -   Run the data_preprocessing.py script to clean and preprocess the
        dataset.

> bash
>
> Copy code
>
> python data_preprocessing.py

2.  **Model Training**:

    -   Run the train_models.py script to train individual and hybrid
        models.

> bash
>
> Copy code
>
> python train_models.py

3.  **Model Evaluation**:

    -   Run the evaluate_models.py script to evaluate and compare the
        models.

> bash
>
> Copy code
>
> python evaluate_models.py

4.  **Visualization**:

    -   Visualizations and performance metrics are saved in the results/
        directory.

**Data**

The Financial Sentiment Analysis dataset used in this project is sourced
from Kaggle, consisting of 5,842 rows and 2 columns. The dataset
includes:

-   **Text Data**: Financial news articles, reports, or other relevant
    text documents.

-   **Sentiment Labels**: Categories indicating the sentiment---Neutral,
    Positive, or Negative.

**Dataset Link**: [[Financial Sentiment Analysis
Dataset]{.underline}](https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis)

**Contributing**

Contributions are welcome! To contribute:

1.  Fork the repository.

2.  Create a new branch (git checkout -b feature-branch).

3.  Make your changes and commit them (git commit -m \'Add feature\').

4.  Push to the branch (git push origin feature-branch).

5.  Create a pull request.

**License**

This project is licensed under the MIT License. See the
[[LICENSE]{.underline}](LICENSE) file for more details.
