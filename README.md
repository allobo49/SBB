# Get to know French4U: Revolutionary Tech to Master French
Welcome to SBB LogoRank, a startup adventure initiated by Anna Dovha and Ana Llorens, two university friends who aim to transform how individuals engage with and master French. 

🇫🇷 **The Challenge**: The journey to fluency in French involves engaging with a wide range of texts that gradually expand one's vocabulary, comprehension, and comfort with the language. However, without guidance, learners can easily find themselves either bored with overly simple texts or frustrated by complex literature beyond their current understanding. This challenge poses a significant barrier to effective and enjoyable language learning.

💡 **Our Solution**: To address this challenge, we have developed an innovative model called "French4U" specifically designed for English speakers learning French. Our technology predicts the difficulty level of French-written texts, which empowers learners to choose reading materials that are perfectly aligned with their learning stage. This targeted approach ensures that every reading experience is both manageable and slightly challenging, promoting faster and more sustainable learning.

Next, we will walk you through the comprehensive steps, methodologies, and outcomes that shaped our journey to creating the final model, French4U. Are you ready to join us on this adventure?

## Methodologies and Results
### 1️⃣ Imports, Data Cleaning and Data Augmentation
The journey begins with the essential setup: importing libraries, data files, and other resources crucial for advancing to subsequent stages.

Among the imported data files, the 'training data' dataset is pivotal as it is used to train the models. However, before embarking on model development, we first go through a data cleaning and augmentation process:

**Data Cleaning**: While reviewing the sentences in the 'training data' dataset, we noticed that some lacked punctuation marks at the end. Recognizing the importance of punctuation for our upcoming data augmentation phase, we ensured that every sentence was properly punctuated:
- *"add_dot"*: Function designed to ensure that a given sentence ends with a punctuation mark. 

**Data Augmentation**: During this stage, we identified and added new features to the 'training data' dataset to further enhance sentence characterization. These features include:

- *"count_words"*: Number of words in a given sentence.
- *"avg_word_length"*: Average length of words in a given sentence.
- *"count_punctuation"*: Number of punctuation marks in a given sentence.
- *"stopword_proportion"*: Proportion of stop words (number of stop words/number of total words) in a given sentence.
- *"flesch_kincaid_readability"*: Readability score of a sentence using the Flesch-Kincaid Grade Level formula.
- *"get_pos_tags"* and *"analyze_pos"*: Analysis of the types of words comprising each sentence and assignment of part-of-speech (POS) tags to each word. For each tag, a new column is created in the dataset, and the corresponding value records the count of words with that specific POS tag present in each sentence.

These features will serve as additional input data alongside the following models. We will compare the results of models trained with these comprehensive features to those trained solely with the sentence as an input.

### 2️⃣ Logistic Regression
Our initial approach to predicting text difficulty levels involves employing logistic regression paired with TF-IDF vectorization. Like the subsequent methods we explore, this strategy has been implemented through two distinct evaluations to develop a model adept at accurately assessing the difficulty of texts written in French for English speakers. These evaluations are structured to assess and compare the efficacy of utilizing varied sets of input features for model training

*Sentence-Only Assessment:* This assessment involves training a model using only the sentence data as input (X). 

*Additional Features Assessment:* In this assessment, we broaden our methodology to incorporate additional linguistic and stylistic features into the analysis, in particular, those developed during the Data Augmentation phase previously described.

Here’s a step-by-step overview of the approach:

**1. Data Preparation and Label Encoding**
Firstly, we encode the 'difficulty' labels of our training data using LabelEncoder, transforming them into a machine-readable format. This step ensures that our model can effectively understand and process the target labels.

**2. Feature Selection and Model Setup**
We define our features (X) according to the type of assessment (sentence-only or with additional features) and our target (y) as the encoded difficulty levels. For text preparation, we employ the TF-IDF Vectorizer to convert text into a format emphasizing key distinguishing words for classification. Additionally, we use Standard Scaling to normalize the numeric values of the additional features. Both text and numeric data are integrated and processed using a ColumnTransformer, ensuring comprehensive and effective utilization for modeling.

**3. Model Pipeline Configuration**
In the sentence-only assessment, we implement a pipeline that includes TF-IDF vectorization followed by a logistic regression classifier. This configuration efficiently bridges the gap from data transformation to model training. Conversely, when the input includes additional features, we utilize the previously described ColumnTransformer for preprocessing. This is complemented by the LogisticRegression classifier, which serves as the final step in the pipeline, ensuring a cohesive and effective approach to handling mixed data types.

**4. Hyperparameter Tuning**
To optimize our model, we employ GridSearchCV with a range of parameters for TF-IDF and logistic regression:

- *"preprocessor__tfidf__ngram_range"*: This parameter determines the range of n-grams to be used for TF-IDF vectorization. An n-gram is a contiguous sequence of n items from a given sample of text or speech. For example, in text, an n-gram might be a sequence of words or letters. Values assessed: (1,1) and (1,2).
- *"preprocessor__tfidf__use_idf"*: This boolean parameter determines whether to use inverse document frequency (IDF) component in the TF-IDF vectorization. Values assessed: True and False.
- *"classifier__C"*: This is the inverse of regularization strength in logistic regression. Regularization is applied to avoid overfitting by penalizing larger coefficients. Values assessed: 0.1, 1 and 10.
- *"classifier__penalty"*: This parameter specifies the norm used in the penalization (regularization) of the logistic regression model. Values assessed: l2 and none.

This extensive search help identifying the best combination of parameters for our model based on accuracy and by performing a 3-fold cross-validation.

**5. Training and Testing**
We split our dataset into training and testing sets to validate the effectiveness of our model. After training, we identify the best model parameters that lead to the highest cross-validation accuracy.

**6. Model Evaluation and Results**
Finally, we evaluate our best model on the test set to measure its performance. The classification reports obtained are the following ones: 

(1) *Classification Report: Only-sentence Assessment's Logistic Regression Best Model *

|                 | Precision | Recall | F1-Score | Support |
|-----------------|-----------|--------|----------|---------|
| **0 (A1 Level)**| 0.58      | 0.60   | 0.59     | 153     |
| **1 (A2 Level)**| 0.49      | 0.45   | 0.47     | 156     |
| **2 (B1 Level)**| 0.39      | 0.44   | 0.41     | 153     |
| **3 (B2 Level)**| 0.55      | 0.35   | 0.43     | 173     |
| **4 (C1 Level)**| 0.47      | 0.39   | 0.43     | 166     |
| **5 (C2 Level)**| 0.46      | 0.69   | 0.55     | 159     |
| **accuracy**    |           |        | 0.48     | 960     |
| **macro avg**   | 0.49      | 0.49   | 0.48     | 960     |
| **weighted avg**| 0.49      | 0.48   | 0.48     | 960     |

Best parameters associated with this model:  {'classifier__C': 10, 'classifier__penalty': 'l2', 'classifier__solver': 'lbfgs', 'tfidf__ngram_range': (1, 2), 'tfidf__use_idf': True}

(2) *Classification Report: Additional Features Assessment's Logistic Regression Best Model *

|                 | Precision | Recall | F1-Score | Support |
|-----------------|-----------|--------|----------|---------|
| **0 (A1 Level)**| 0.58      | 0.71   | 0.64     | 153     |
| **1 (A2 Level)**| 0.47      | 0.44   | 0.46     | 156     |
| **2 (B1 Level)**| 0.49      | 0.50   | 0.50     | 153     |
| **3 (B2 Level)**| 0.48      | 0.51   | 0.49     | 173     |
| **4 (C1 Level)**| 0.49      | 0.37   | 0.42     | 166     |
| **5 (C2 Level)**| 0.55      | 0.54   | 0.55     | 159     |
| **accuracy**    |           |        | 0.51     | 960     |
| **macro avg**   | 0.51      | 0.51   | 0.51     | 960     |
| **weighted avg**| 0.51      | 0.51   | 0.51     | 960     |

Best parameters associated with this model:  {'classifier__C': 10, 'classifier__penalty': 'l2', 'preprocessor__tfidf__ngram_range': (1, 1), 'preprocessor__tfidf__use_idf': True}

**7. Conclusion**

The inclusion of additional features alongside the basic TF-IDF vectorization of the sentences appears to enhance the model's ability to more accurately and effectively classify the difficulty levels of French texts for English speakers. This is particularly evident in the improved accuracy (0.51 in the Additional Features Assessment VS 0.48 in the Only-Sentence Assessment) and macro averages, which suggest that the model benefits from a richer set of input data. These enhancements likely provide the model with a more nuanced understanding of the text, improving its performance, especially in correctly identifying instances at the extremes of the difficulty spectrum (A1 and C2 levels).

Regarding hyperparameters, both best models agreed that a regularization strength (classifier__C) of 10 was ideal, reflecting a preference for minimal regularization to allow greater flexibility in the model's decision boundaries. This level of regularization suggests that the models are capable of managing the balance between bias and variance without heavily penalizing the size of the coefficients. Additionally, the 'l2' penalty was chosen for both models, endorsing its effectiveness in controlling overfitting by squaring the coefficients, which proves beneficial regardless of the model configuration. Another shared setting was the use of inverse document frequency (IDF) in the TF-IDF vectorization, with both models performing optimally with IDF enabled. This setting underlines the importance of reducing the influence of frequently occurring terms, thereby enhancing the overall discriminatory power of the model's text processing.

However, the models diverged on the n-gram range used in TF-IDF vectorization. The Only Sentence model found that a range of (1, 2) — encompassing both unigrams and bigrams — was most effective, suggesting that the inclusion of bigrams provided significant contextual benefit when relying solely on textual data. In contrast, the Additional Features model opted for (1, 1), relying exclusively on unigrams. This indicates that the presence of additional numeric and categorical data diminished the need for capturing broader textual contexts through bigrams, as these contexts were adequately supplemented by the other features.

### 3️⃣ Random Forest Classifier
The procedure for implementing the Random Forest classifier closely mirrors that of the Logistic Regression, with a few key distinctions:

**Pipeline configuration:** In the pipeline, the Logistic Regression classifier is replaced by a RandomForestClassifier. Random Forest is an ensemble learning method based on decision trees, where multiple trees are generated during the training process, and their results are aggregated to provide the final output. This method is particularly noted for its robustness against overfitting and its ability to handle both linear and non-linear relationships.

**Parameter tuning specifics:** To enhance our model's performance, we once more utilize GridSearchCV, employing a 5-fold cross-validation strategy (as it offers a lower computational burden per fold compared to the previous approach) and some different parameters for TF-IDF and Random Forest Classifier:

- *"preprocessor__tfidf__ngram_range"*: This parameter is identical to the one specified for Logistic Regression. Values assessed: (1,1) and (1,2).
- *"preprocessor__tfidf__use_idf"*: This parameter is identical to the one specified for Logistic Regression. Values assessed: True and False.
- *"classifier__n_estimators"*: This parameter determines the number of decision trees in the random forest. Values assessed: 100, 200 and 300.
- *"classifier__max_depth"*: It sets the maximum depth of each decision tree in the forest. Values assessed: None, 10, 20 and 30.
- *"classifier__min_samples_split"*: This parameter specifies the minimum number of samples required to split an internal node. Values assessed: 2, 5 and 10.
- *"classifier__min_samples_leaf"*: It sets the minimum number of samples required to be at a leaf node. Values assessed: 1, 2 and 4.

**Results**

(1) *Classification Report: Only-sentence Assessment's Random Forest Best Model *

|                 | Precision | Recall | F1-Score | Support |
|-----------------|-----------|--------|----------|---------|
| **0 (A1 Level)**| 0.41      | 0.78   | 0.54     | 153     |
| **1 (A2 Level)**| 0.35      | 0.29   | 0.32     | 156     |
| **2 (B1 Level)**| 0.40      | 0.36   | 0.38     | 153     |
| **3 (B2 Level)**| 0.39      | 0.34   | 0.36     | 173     |
| **4 (C1 Level)**| 0.33      | 0.27   | 0.30     | 166     |
| **5 (C2 Level)**| 0.49      | 0.36   | 0.41     | 159     |
| **accuracy**    |           |        | 0.40     | 960     |
| **macro avg**   | 0.40      | 0.40   | 0.39     | 960     |
| **weighted avg**| 0.40      | 0.40   | 0.38     | 960     |

Best parameters associated with this model: {'classifier__max_depth': None, 'classifier__min_samples_leaf': 1, 'classifier__min_samples_split': 5, 'classifier__n_estimators': 300, 'tfidf__ngram_range': (1, 1), 'tfidf__use_idf': True}

(2) *Classification Report: Additional Features Assessment's Random Forest Best Model*

|                 | Precision | Recall | F1-Score | Support |
|-----------------|-----------|--------|----------|---------|
| **0 (A1 Level)**| 0.55      | 0.77   | 0.64     | 153     |
| **1 (A2 Level)**| 0.41      | 0.40   | 0.41     | 156     |
| **2 (B1 Level)**| 0.39      | 0.41   | 0.40     | 153     |
| **3 (B2 Level)**| 0.40      | 0.32   | 0.36     | 173     |
| **4 (C1 Level)**| 0.38      | 0.36   | 0.37     | 166     |
| **5 (C2 Level)**| 0.51      | 0.43   | 0.47     | 159     |
| **accuracy**    |           |        | 0.45     | 960     |
| **macro avg**   | 0.44      | 0.45   | 0.44     | 960     |
| **weighted avg**| 0.44      | 0.45   | 0.44     | 960     |

Best parameters associated with this model:  {'classifier__max_depth': None, 'classifier__min_samples_leaf': 1, 'classifier__min_samples_split': 5, 'classifier__n_estimators': 300, 'preprocessor__tfidf__ngram_range': (1, 1), 'preprocessor__tfidf__use_idf': True}

**Conclusion**

The Random Forest model trained on the "Only-sentence Assessment" achieved an accuracy of 40%, demonstrating moderate precision, recall, and F1-scores across proficiency levels. Notably, it showed better performance in distinguishing lower proficiency levels (A1, A2) but still indicated room for improvement in overall accuracy.

In contrast, the Random Forest model trained on the "Additional Features Assessment" outperformed its counterpart, achieving an accuracy of 45% with improved precision, recall, and F1-scores across all proficiency levels. Particularly, this model excelled in accurately classifying beginner proficiency levels (A1), showcasing the effectiveness of incorporating diverse features in proficiency assessment tasks.

Regarding the optimal hyperparameters, which turned out to be identical in both models, they represent a balanced trade-off between model complexity and generalization. With n_estimators set to 300, the models benefit from a larger ensemble size, enhancing stability and reducing overfitting. Setting max_depth to None allows trees to grow without restriction, capturing complex patterns. min_samples_split and min_samples_leaf values of 5 and 1, respectively, ensure robust splitting criteria and sufficient samples at leaf nodes for accurate predictions. Additionally, using unigrams (ngram_range=(1, 1)) with IDF weighting (use_idf=True) optimizes feature representation, emphasizing informative features while mitigating the impact of common terms. 

### 4️⃣ Decision Tree Classifier

The procedure for implementing the Decision Tree Classifier is also almost equal to that of the Logistic Regression, with a few key distinctions:

**Pipeline configuration:** In the pipeline, the Logistic Regression classifier is replaced by a DecisionTreeClassifier, which is a supervised learning algorithm used for classification tasks.

**Parameter tuning specifics:** To enhance our model's performance, we utilize GridSearchCV once more, employing a 5-fold cross-validation and the identical parameters as before for TF-IDF and the Decision Tree Classifier. However, we exclude "classifier__n_estimators", as it's not applicable in this scenario.

**Results**

(1) *Classification Report: Only-sentence Assessment's Decision Tree Best Model *

|                 | Precision | Recall | F1-Score | Support |
|-----------------|-----------|--------|----------|---------|
| **0 (A1 Level)**| 0.34      | 0.62   | 0.44     | 153     |
| **1 (A2 Level)**| 0.26      | 0.23   | 0.24     | 156     |
| **2 (B1 Level)**| 0.18      | 0.14   | 0.16     | 153     |
| **3 (B2 Level)**| 0.30      | 0.40   | 0.34     | 173     |
| **4 (C1 Level)**| 0.32      | 0.17   | 0.22     | 166     |
| **5 (C2 Level)**| 0.41      | 0.19   | 0.24     | 159     |
| **accuracy**    |           |        | 0.29     | 960     |
| **macro avg**   | 0.28      | 0.29   | 0.28     | 960     |
| **weighted avg**| 0.29      | 0.29   | 0.28     | 960     |

Best parameters associated with this model: {'classifier__max_depth': 20, 'classifier__min_samples_leaf': 1, 'classifier__min_samples_split': 2, 'tfidf__ngram_range': (1, 1), 'tfidf__use_idf': True}

(2) *Classification Report: Additional Features Assessment's Decision Tree Best Model*

|                 | Precision | Recall | F1-Score | Support |
|-----------------|-----------|--------|----------|---------|
| **0 (A1 Level)**| 0.57      | 0.66   | 0.61     | 153     |
| **1 (A2 Level)**| 0.42      | 0.50   | 0.46     | 156     |
| **2 (B1 Level)**| 0.28      | 0.28   | 0.28     | 153     |
| **3 (B2 Level)**| 0.36      | 0.28   | 0.31     | 173     |
| **4 (C1 Level)**| 0.35      | 0.27   | 0.30     | 166     |
| **5 (C2 Level)**| 0.40      | 0.45   | 0.42     | 159     |
| **accuracy**    |           |        | 0.40     | 960     |
| **macro avg**   | 0.40      | 0.41   | 0.40     | 960     |
| **weighted avg**| 0.39      | 0.40   | 0.40     | 960     |

Best parameters associated with this model: {'classifier__max_depth': 10, 'classifier__min_samples_leaf': 1, 'classifier__min_samples_split': 10, 'preprocessor__tfidf__ngram_range': (1, 1), 'preprocessor__tfidf__use_idf': False}

**Conclusion**

The Decision Tree model trained on the "Only-sentence Assessment" achieved low-moderate accuracy (29%) with varying precision, recall, and F1-scores across proficiency levels. In contrast, the "Additional Features Assessment" model significantly improved classification performance, achieving an accuracy of 40% and higher precision, recall, and F1-scores across all proficiency levels of French.

This improvement suggests again that incorporating additional features enhances the model's ability to accurately classify proficiency levels. Notably, the "Additional Features Assessment" model excelled in classifying beginner proficiency levels, demonstrating the importance of diverse features in proficiency assessment tasks.

Regarding the optimal hyperparameters, in the "Only-sentence Assessment" model, a deeper tree structure (max_depth=20) and less strict splitting criteria (min_samples_split=2) were chosen to potentially capture intricate patterns. Conversely, the "Additional Features Assessment" model opted for a shallower tree (max_depth=10) with stricter splitting criteria (min_samples_split=10) to prevent overfitting. Both models employed unigram TF-IDF representation, emphasizing single words' importance across documents. These hyperparameters were carefully tailored to each feature assessment's characteristics, highlighting the models' adaptability and the importance of parameter tuning for optimal classification performance.

### 5️⃣ KNN Classifier

The process of implementing the KNN classifier closely resembles that of Logistic Regression too, with a few key distinctions:

**Pipeline configuration:** In the pipeline, the Logistic Regression classifier is substituted with a KNN (K-Nearest Neighbors) classifier. KNN is a simple yet effective classification algorithm that classifies new data points based on the majority class of their neighboring data points.

**Parameter tuning specifics:** To optimize our model, we employ GridSearchCV again, with a 5-fold cross-validation and some different parameters for TF-IDF and KNN Classifier:

- *"tfidf__ngram_range"*: This parameter matches the one specified for Logistic Regression as *"preprocessor__tfidf__ngram_range"*. Values assessed: (1,1) and (1,2).
- *"tfidf__use_idf"*: This parameter matches the one specified for Logistic Regression as *"preprocessor__tfidf__use_idf"*. Values assessed: True and False.
- *"classifier__n_neighbors"*: This parameter determines the number of neighbors to consider for classification. KNN (K-Nearest Neighbors) assigns a class label to a new data point based on the majority class among its n_neighbors nearest neighbors. Values assessed: 3, 5, 6 (equal to the number of labels) and 7.
- *"classifier__weights"*: This parameter specifies the weight function used in prediction. It can take two values: 'uniform' (all neighbors are weighted equally in the prediction process) and 'distance'(closer neighbors are given more weight in the prediction, with weights inversely proportional to their distance from the query point).
- *"classifier__algorithm"*: This parameter specifies the algorithm used to compute the nearest neighbors. It can take four values: 'auto'(automatically selects the most appropriate algorithm based on the training data), 'ball_tree' (uses a Ball Tree data structure to perform nearest neighbor search efficiently), 'kd_tree' (uses a KD Tree data structure for efficient nearest neighbor search) and 'brute' (computes nearest neighbors by brute force, i.e., by comparing the distances to all training samples).

**Results**

(1) *Classification Report: Only-sentence Assessment's KNN Classifier Best Model *

|                 | Precision | Recall | F1-Score | Support |
|-----------------|-----------|--------|----------|---------|
| **0 (A1 Level)**| 0.30      | 0.73   | 0.42     | 153     |
| **1 (A2 Level)**| 0.28      | 0.33   | 0.30     | 156     |
| **2 (B1 Level)**| 0.22      | 0.22   | 0.22     | 153     |
| **3 (B2 Level)**| 0.46      | 0.24   | 0.32     | 173     |
| **4 (C1 Level)**| 0.61      | 0.23   | 0.33     | 166     |
| **5 (C2 Level)**| 0.59      | 0.37   | 0.46     | 159     |
| **accuracy**    |           |        | 0.35     | 960     |
| **macro avg**   | 0.41      | 0.35   | 0.34     | 960     |
| **weighted avg**| 0.41      | 0.35   | 0.34     | 960     |

Best parameters associated with this model: {'classifier__algorithm': 'auto', 'classifier__n_neighbors': 3, 'classifier__weights': 'distance', 'tfidf__ngram_range': (1, 1), 'tfidf__use_idf': True}

(2) *Classification Report: Additional Features Assessment's KNN Classifier Best Model*

|                 | Precision | Recall | F1-Score | Support |
|-----------------|-----------|--------|----------|---------|
| **0 (A1 Level)**| 0.50      | 0.71   | 0.59     | 153     |
| **1 (A2 Level)**| 0.38      | 0.38   | 0.38     | 156     |
| **2 (B1 Level)**| 0.33      | 0.26   | 0.29     | 153     |
| **3 (B2 Level)**| 0.34      | 0.29   | 0.31     | 173     |
| **4 (C1 Level)**| 0.32      | 0.29   | 0.30     | 166     |
| **5 (C2 Level)**| 0.36      | 0.39   | 0.38     | 159     |
| **accuracy**    |           |        | 0.38     | 960     |
| **macro avg**   | 0.37      | 0.39   | 0.37     | 960     |
| **weighted avg**| 0.37      | 0.38   | 0.37     | 960     |

Best parameters associated with this model: {'classifier__algorithm': 'auto', 'classifier__n_neighbors': 7, 'classifier__weights': 'distance', 'preprocessor__tfidf__ngram_range': (1, 1), 'preprocessor__tfidf__use_idf': False}

**Conclusion**

In conclusion, the KNN classifiers trained on two different feature assessments showed varying degrees of success in accurately classifying proficiency levels. Despite employing optimal hyperparameters and a thorough evaluation process, both models struggled with precision and recall, particularly for lower proficiency levels (A1, A2).

The "Additional Features Assessment" model exhibited slightly better performance compared to the "Only-sentence Assessment" model (0.35 vs 0.38), achieving higher accuracy and precision across most proficiency levels. This suggests that the inclusion of additional features contributed to marginal improvements in classification accuracy. However, both models still faced challenges in accurately distinguishing between different proficiency levels, indicating the need for further optimization and feature engineering.

Regarding optimal hypermeparameters, they turned out to be almost identical in both models except from *"classifier__n_neighbors"*. In the "Only-sentence Assessment" dataset, a smaller neighborhood size of 3 neighbors sufficed. This suggests that the feature space may be less complex, with neighboring samples providing sufficient contextual information for classification. Conversely, the "Additional Features Assessment" dataset incorporated additional and more diverse features, introducing greater variability and complexity to the feature space. As a result, a larger neighborhood size of 7 neighbors was necessary to capture the broader spectrum of feature interactions and relationships. This larger neighborhood enables the model to consider a wider array of samples during classification.

### 6️⃣ Neural Networks
### 7️⃣ FlauBERT Model
### 8️⃣ 
### 🖇️ Collective Overview
## Best Model: FlauBERT Model
## Ranking
## Streamlit Application
## Video
## Contributions
