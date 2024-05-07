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

> All the hyperparameter values mentioned in this document have been chosen with the intent of achieving optimal performance while also taking computational costs into account.

This extensive search help identifying an optimal combination of parameters for our model based on accuracy and by performing a 3-fold cross-validation.

**5. Training and Testing**
We split our dataset into training and testing sets to validate the effectiveness of our model. After training, we identify the best model parameters that lead to the highest cross-validation accuracy.

**6. Model Evaluation and Results**
Finally, we evaluate our best model on the test set to measure its performance. The classification reports obtained are the following ones: 

(1) *Classification Report: Only-sentence Assessment's Logistic Regression Best Model*

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

(2) *Classification Report: Additional Features Assessment's Logistic Regression Best Model*

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

(1) *Classification Report: Only-sentence Assessment's Random Forest Best Model*

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

(1) *Classification Report: Only-sentence Assessment's Decision Tree Best Model*

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

(1) *Classification Report: Only-sentence Assessment's KNN Classifier Best Model*

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

In pursuit of higher accuracy levels, we have chosen to explore novel methodologies such as neural networks. This machine learning technique involves a series of steps, outlined as follows:

**1. Label Encoding:**
The 'difficulty' labels are encoded using a LabelEncoder to convert them into numerical format, suitable for model training.

**2. TF-IDF Vectorization:**
The sentences are transformed into TF-IDF (Term Frequency-Inverse Document Frequency) vectors using TfidfVectorizer. This step converts text data into numerical features, where each word's importance is weighted based on its frequency in the sentence and rarity across all sentences.

**3. Train-Test Split:**
The dataset is split into training and testing sets using train_test_split from sklearn.model_selection. This separation ensures that the model's performance can be evaluated on unseen data.

**4. Feature Scaling:**
The input features are scaled using MinMaxScaler (sentences) and StandardScaler (features) to ensure that all features are on a similar scale. This prevents certain features from dominating others during training.

**5. Convert to Torch Tensors:**
The scaled feature vectors and labels are converted into PyTorch tensors using torch.tensor. This step is necessary for compatibility with PyTorch's neural network framework.

**6. Define Neural Network Architecture:**
A neural network class NeuralNetwork is defined using PyTorch's nn.Module. It consists of two linear layers (nn.Linear) with ReLU activation function (nn.ReLU), defining the forward pass of the network.

**7. Hyperparameters Initialization:**
Hyperparameters such as the number of iterations and learning rates are defined. These parameters control the training process and optimization of the neural network:

- *"iterations"*: This hyperparameter determines the number of times the entire dataset is passed forward and backward through the neural network during the training process. It represents the number of times the model updates its parameters to minimize the loss function. Values assessed: 500, 1000, 1500.
- *"learning_rates"*:This is the hyperparameter that controls the step size at each iteration while moving toward a minimum of the loss function. It determines how much the model's parameters are adjusted in each iteration of the optimization algorithm. Values assessed: 0.001, 1.049 and 12.031.

**8. Model Training and Evaluation:**
The model is trained and evaluated using a nested loop over different combinations of iterations and learning rates. The neural network is trained using stochastic gradient descent (torch.optim.SGD) and cross-entropy loss (nn.CrossEntropyLoss).

**9. Select the Best Model and Generate Classification Report:**
The model with the highest accuracy on the test set is saved as the best model, which is used to generate predictions on the test set, and a classification report is generated using classification_report from sklearn.metrics. This report provides metrics such as precision, recall, and F1-score for each class, enabling a comprehensive evaluation of the model's performance.

**10. Results:**

(1) *Classification Report: Only-sentence Assessment's Neural Network Best Model*

|                 | Precision | Recall | F1-Score | Support |
|-----------------|-----------|--------|----------|---------|
| **0 (A1 Level)**| 0.55      | 0.57   | 0.56     | 166     |
| **1 (A2 Level)**| 0.37      | 0.38   | 0.37     | 158     |
| **2 (B1 Level)**| 0.39      | 0.35   | 0.37     | 166     |
| **3 (B2 Level)**| 0.39      | 0.36   | 0.37     | 153     |
| **4 (C1 Level)**| 0.39      | 0.41   | 0.40     | 152     |
| **5 (C2 Level)**| 0.44      | 0.46   | 0.45     | 165    |
| **accuracy**    |           |        | 0.42     | 960     |
| **macro avg**   | 0.42      | 0.42   | 0.42     | 960     |
| **weighted avg**| 0.42      | 0.42   | 0.42     | 960     |

Best parameters associated with this model: {Iterations 1500, Learning Rate 1.049, Accuracy 42.29%}

(2) *Classification Report: Additional Features Assessment's Neural Network Best Model*

|                 | Precision | Recall | F1-Score | Support |
|-----------------|-----------|--------|----------|---------|
| **0 (A1 Level)**| 0.80      | 0.23   | 0.36     | 153     |
| **1 (A2 Level)**| 0.40      | 0.78   | 0.53     | 156     |
| **2 (B1 Level)**| 0.50      | 0.43   | 0.46     | 153     |
| **3 (B2 Level)**| 0.47      | 0.50   | 0.49     | 173     |
| **4 (C1 Level)**| 0.38      | 0.37   | 0.38     | 166     |
| **5 (C2 Level)**| 0.53      | 0.45   | 0.48     | 159     |
| **accuracy**    |           |        | 0.46     | 960     |
| **macro avg**   | 0.51      | 0.46   | 0.45     | 960     |
| **weighted avg**| 0.51      | 0.46   | 0.45     | 960     |

Best parameters associated with this model: {Iterations 1500, Learning Rate 1.049, Accuracy 46.15%}

**Conclusion**

In comparing the performance of the neural network models for the "Only-sentence Assessment" and "Additional Features Assessment," notable differences emerge.

The "Only-sentence Assessment" model achieved an overall accuracy of 42.29%, with precision, recall, and F1-scores ranging from 0.37 to 0.56 across different difficulty levels. This model demonstrated a balanced performance across most classes.

Conversely, the "Additional Features Assessment" model exhibited an improved overall accuracy of 46.15%. Notably, this model displayed higher precision, recall, and F1-scores for several difficulty levels, particularly for classes 0 (A1 Level) and 1 (A2 Level). However, it demonstrated lower performance for classes 3 (B2 Level) and 5 (C2 Level).

Overall, while both models utilized the same hyperparameters (Iterations: 1500, Learning Rate: 1.049), the inclusion of additional features in the latter model led to a modest improvement in accuracy and performance metrics.

### 7️⃣ Best Model: FlauBERT Model

Not fully satisfied with the obtained accuracy levels, we decided to explore alternative methodologies, in this case, the FlauBERT Model from [Hugging Face](https://huggingface.co/docs/transformers/en/model_doc/flaubert), as described:

> The FlauBERT model was proposed in the paper FlauBERT: Unsupervised Language Model Pre-training for French by Hang Le et al. It’s a transformer model pretrained using a masked language modeling (MLM) objective (like BERT).

For the FlauBERT model, it is possible to choose from several options:

| Model name                | Number of layers | Attention Heads | Embedding Dimension | Total Parameters |
|---------------------------|------------------|-----------------|---------------------|------------------|
| flaubert-small-cased      | 6                | 8               | 512                 | 54M              |
| flaubert-base-uncased     | 12               | 12              | 768                 | 137M             |
| flaubert-base-cased       | 12               | 12              | 768                 | 138M             |
| flaubert-large-cased      | 24               | 16              | 1024                | 373M             |

We decided to use the **FlauBERT large cased model** because it has greater depth, a more sophisticated attention mechanism, a larger embedding size, and a higher parameter count. Larger models such as the FlauBERT large cased typically outperform smaller ones across a variety of language understanding benchmarks, potentially offering higher accuracy.

Also, the [BERT authors recommend fine-tuning](https://github.com/google-research/bert) for four epochs with the following hyperparameter options:

- batch sizes: 8, 16, 32, 64, 128
- learning rates: 3e-4, 1e-4, 5e-5, 3e-5

Given the computational limitation, we will train our model only on 2 different different batch sizes: 16 and 32.

**FlauBERT Model over different learning rates**

Important to note that our first model is trained using the **AdamW optimizer**, which is a variant of the traditional Adam optimizer. AdamW incorporates a regularization technique known as [weight decay](https://github.com/tml-epfl/why-weight-decay), which is used in training neural networks to prevent overfitting. It functions by incorporating a term into the loss function that penalizes large weights.

Moreover, we also employ a **Linear Learning Rate Scheduler** to manage the learning rate throughout the training process. This scheduler starts with a relatively high learning rate and gradually decreases it to near zero by the end of the training. This approach ensures that we begin training with aggressive learning steps and fine-tune the model parameters more delicately as training progresses.

Although this training setup does not include a warm-up phase where the learning rate would gradually ramp up before decreasing, the scheduler is configured to reduce the learning rate slightly with each training step. This gradual reduction helps in stabilizing the training as it advances.

We would like to first observe the results with this setup and may adjust these parameters further based on the outcomes of the initial training phase. Here are the training results using a batch size of 16 across various learning rates:

| Learning Rate | Epoch | Average Loss | Accuracy   | Precision | Recall    | F1 Score  | Notes                  |
|---------------|-------|--------------|------------|-----------|-----------|-----------|------------------------|
| 0.0001        | 1     | 0.15489      | 22.71%     | 18.81%    | 22.71%    | 12.83%    | -                      |
| 0.0001        | 2     | 0.10375      | 37.92%     | 46.40%    | 37.92%    | 32.24%    | -                      |
| 0.0001        | 3     | 0.09681      | 45.62%     | 52.56%    | 45.62%    | 42.65%    | -                      |
| 0.0001        | 4     | 0.07292      | 51.15%     | 50.92%    | 51.15%    | 50.50%    | -                      |
| 5e-05         | 1     | 0.10961      | 42.92%     | 39.80%    | 42.92%    | 39.47%    | -                      |
| 5e-05         | 2     | 0.08257      | 50.42%     | 53.07%    | 50.42%    | 48.79%    | -                      |
| 5e-05         | 3     | 0.06633      | 55.31%     | 57.52%    | 55.31%    | 54.84%    | -                      |
| 5e-05         | 4     | 0.05221      | 56.98%     | 57.05%    | 56.98%    | 56.76%    | Best overall performance |
| 3e-05         | 1     | 0.10324      | 44.58%     | 52.58%    | 44.58%    | 42.50%    | -                      |
| 3e-05         | 2     | 0.07793      | 48.96%     | 49.55%    | 48.96%    | 46.42%    | -                      |
| 3e-05         | 3     | 0.06602      | 55.62%     | 56.60%    | 55.62%    | 55.01%    | -                      |
| 3e-05         | 4     | 0.05735      | 55.94%     | 55.52%    | 55.94%    | 55.39%    | -                      |
| 2e-05         | 1     | 0.10434      | 44.90%     | 50.87%    | 44.90%    | 42.50%    | -                      |
| 2e-05         | 2     | 0.07980      | 49.58%     | 50.54%    | 49.58%    | 48.43%    | -                      |
| 2e-05         | 3     | 0.06963      | 53.44%     | 53.24%    | 53.44%    | 52.64%    | -                      |
| 2e-05         | 4     | 0.06303      | 52.71%     | 52.48%    | 52.71%    | 52.28%    | -                      |

Our training was interrupted, preventing a complete evaluation of the results for a batch size of 32. However, based on our observations, a learning rate of 5e-05 produced the highest performance across accuracy, precision, recall, and F1 score (surpassing all previous models). Consequently, we decided to continue training using this learning rate and adjust the batch size to thoroughly assess the results for batch 32.

We further trained the model on the batch size 32 over 4 epochs with the learning rate 5e-05 (previous highest performance in terms of accuracy). These are the results obtained:

| Epoch | Learning Rate | Average Loss     | Accuracy   | Precision | Recall   | F1 Score  |
|-------|---------------|------------------|------------|-----------|----------|-----------|
| 1/4   | 5e-5          | 0.053033780585974| 0.43125    | 0.5486596 | 0.43125  | 0.4053166 |
| 2/4   | 5e-5          | 0.040040116741632| 0.4791667  | 0.5172350 | 0.4791667| 0.4742365 |
| 3/4   | 5e-5          | 0.031953962224846| 0.5510417  | 0.5744112 | 0.5510417| 0.5477183 |
| 4/4   | 5e-5          | 0.025974183475288| 0.5739583  | 0.5810352 | 0.5739583| 0.5750711 |

The main difference between the performance of the training on the batch 16 and 32 with the same learning rate 5e-5 is the average loss. From the graph, it's clear that the average loss for the batch size of 32 is significantly lower than that for the batch size of 16 at every epoch.

![16 and 32 batch_loss.png](https://github.com/AnyaLang/SBB_ML/blob/125ad85ffe0c16a54ce6138367cc1b7af2dc0b6e/16%20and%2032%20batch_loss.png)

#### **Model configured with a learning rate of 5e-5, a batch size of 32, and trained for 6 epochs.**

Based on the previous results, we opted to train the model over a larger number of epochs, specifically over 6 epochs, using a batch size of 32 and maintaining the previous learning rate of 5e-5. Here are the results obtained: 

| Epoch | Learning Rate | Average Loss   | Accuracy   | Precision   | Recall   | F1 Score   |
|-------|---------------|----------------|------------|-------------|----------|------------|
| 1/6   | 5e-5          | 0.05631464881  | 0.4041667  | 0.4795025   | 0.4041667| 0.3879119  |
| 2/6   | 5e-5          | 0.04334156585  | 0.5333333  | 0.5451042   | 0.5333333| 0.5311008  |
| 3/6   | 5e-5          | 0.03416992826  | 0.546875   | 0.5649137   | 0.546875 | 0.5387779  |
| 4/6   | 5e-5          | 0.02753307774  | 0.6010417  | 0.6019167   | 0.6010417| 0.5942725  |
| 5/6   | 5e-5          | 0.02137682571  | 0.590625   | 0.5990724   | 0.590625 | 0.5874222  |
| 6/6   | 5e-5          | 0.01756872524  | 0.596875   | 0.6022174   | 0.596875 | 0.5978323  |

We then submitted two models on Kaggle, from the epoch 4 and 6. While the epoch 4 had higher accuracy, it provided the result of 0.573 on Kaggle. **For the model on the epoch 6th, the F1 score was higher, leading to the result of Kaggle of 0.601.** This demonstrates that relying solely on accuracy might not give a comprehensive assessment of a model's performance. Therefore, it is crucial to consider multiple metrics.

We also experimented and changed the number of epochs to 4, 6 and 8. However, 6 epochs resulted in the highest accuracy of the model and F1 value.

#### **Model configured with a different learning rate adjustement**

Aiming to explore new possibilities that could enhance our model, we decided to make modifications to the Warm-Up Phase and adjust the Learning Rate:

- *Increased the Initial Learning Rate:* We start with a higher initial learning rate of 1e-4.

- *Added Warm-Up Steps:* Introduce a warm-up phase where the learning rate will linearly increase to this higher initial rate over a number of steps. A common strategy is to set the warm-up steps to 10% of the total training steps.

`# Initialize the optimizer with a higher initial learning rate`
`optimizer = AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08)`

`# Scheduler with warm-up phase`
`scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)`

The results for 8 epochs using the adjusted learning rate are as follows:

| Epoch | Learning Rate | Average Loss      | Accuracy   | Precision | Recall   | F1 Score  |
|-------|---------------|-------------------|------------|-----------|----------|-----------|
| 1/8   | 0.00009722    | 0.059470302890986 | 0.365625   | 0.3330621 | 0.365625 | 0.3027832 |
| 2/8   | 0.00008333    | 0.053232380685707 | 0.4791667  | 0.4275987 | 0.4791667| 0.4342403 |
| 3/8   | 0.00006944    | 0.038865594674523 | 0.5541667  | 0.5699164 | 0.5541667| 0.5361277 |
| 4/8   | 0.00005556    | 0.034336493226389 | 0.565625   | 0.5869896 | 0.565625 | 0.5635115 |
| 5/8   | 0.00004167    | 0.022284670624261 | 0.6010417  | 0.5997075 | 0.6010417| 0.5990875 |
| 6/8   | 0.00002778    | 0.015347646844263 | 0.603125   | 0.6056449 | 0.603125 | 0.5911651 |
| 7/8   | 0.00001389    | 0.009685143810930 | 0.6020833  | 0.6107261 | 0.6020833| 0.6032911 |
| 8/8   | 0.00000000    | 0.005019144429146 | 0.6166667  | 0.6212918 | 0.6166667| 0.6177674 |

![learning_rate.png](https://github.com/AnyaLang/SBB_ML/blob/b509447374760d91759c3c62027701d928a15ce2/Model%20with%20a%20different%20learning%20rate%20adjustement.png)

While the model with the adjusted learning rate demonstrated a higher accuracy score and performed better than the models before over other metrics, the submission on Kaggle provided a lower acore. We also adjusted the number of epochs to 15 and lower, however, the results were worse.

#### **Model configured with a batch size of 16 and different learning rates**

We decided, to explore a bit more the training of the models over the lower batch size and different learning rates than before.

Each training session was conducted with a distinct learning rate, ranging from 1e-06 to 2e-05. The goal was to find an optimal rate that balances fast learning without overshooting the minimum of the loss function. For each learning rate, the model was trained over two epochs. This limited exposure was designed to quickly assess the impact of each learning rate without extensive computational cost.

| Learning Rate | Epoch | Average Loss    | Validation Accuracy |
|---------------|-------|-----------------|---------------------|
| 1e-05         | 1/2   | 1.776790196200212 | 0.3614583333333333  |
| 1e-05         | 2/2   | 1.4493117819229762| 0.4354166666666667  |
| 2e-05         | 1/2   | 1.8514792347947757| 0.43020833333333336 |
| 2e-05         | 2/2   | 1.3657454945147038| 0.5104166666666666  |
| 5e-06         | 1/2   | 2.1787539795041084| 0.3072916666666667  |
| 5e-06         | 2/2   | 1.7749672616521517| 0.346875            |
| 1e-06         | 1/2   | 2.78974984139204  | 0.19375             |
| 1e-06         | 2/2   | 2.3164451534549397| 0.20208333333333334 |

We achieved an accuracy of 51% within just two epochs using a learning rate of 2e-05. Encouraged by these results, we have decided to continue refining the model with this learning rate. To explore the model's capacity further, we plan to keep the batch size to 16 and adjust the learning rate to 3e-05, while extending the training period to 15 epochs.

#### **Model configured with a learning rate of 3e-05, a batch size of 16 and over a large number of epochs**

| Epoch  | Learning Rate | Average Loss    | Validation Accuracy | Notes                                     |
|--------|---------------|-----------------|---------------------|-------------------------------------------|
| 1/15   | 3e-05         | 1.6861949928105 | 0.459375            | Saved as `best_model_lr3e-05_ep1_acc0.46.pt`  |
| 2/15   | 3e-05         | 1.2930432051420 | 0.5104167           | Saved as `best_model_lr3e-05_ep2_acc0.51.pt`  |
| 3/15   | 3e-05         | 1.1450499561926 | 0.5020833           |                                           |
| 4/15   | 3e-05         | 0.9551384929568 | 0.5479167           | Saved as `best_model_lr3e-05_ep4_acc0.55.pt`  |
| 5/15   | 3e-05         | 0.8847448159009 | 0.5552083           | Saved as `best_model_lr3e-05_ep5_acc0.56.pt`  |
| 6/15   | 3e-05         | 0.6622620061661 | 0.5541667           |                                           |
| 7/15   | 3e-05         | 0.5362344713571 | 0.5625              | Saved as `best_model_lr3e-05_ep7_acc0.56.pt`  |
| 8/15   | 3e-05         | 0.4089817595979 | 0.5875              | Saved as `best_model_lr3e-05_ep8_acc0.59.pt`  |
| 9/15   | 3e-05         | 0.3382450588358 | 0.5885417           | Saved as `best_model_lr3e-05_ep9_acc0.59.pt`  |
| 10/15  | 3e-05         | 0.2671806021050 | 0.5770833           |                                           |
| 11/15  | 3e-05         | 0.2069165084783 | 0.590625            | Saved as `best_model_lr3e-05_ep11_acc0.59.pt` |
| 12/15  | 3e-05         | 0.1845976701433 | 0.5927083           | Saved as `best_model_lr3e-05_ep12_acc0.59.pt` |
| 13/15  | 3e-05         | 0.1560351345979 | 0.5958333           | Saved as `best_model_lr3e-05_ep13_acc0.60.pt` |
| 14/15  | 3e-05         | 0.1160823275044 | 0.584375            |                                           |
| 15/15  | 3e-05         | 0.1117068582588 | 0.5885417           |                                           |

With this setting, we were able to achieve an accuracy of 0.590 on Kaggle. In the subsequent training session, we achieved an accuracy on Kaggle of 0.593. While we strive to make our code reproducible, some aspects of the model are outside our control and are influenced by a degree of randomness.

`best_model_path = 'best_model_lr3e-05_ep13_acc0.60.pt'  #the second time we run the code, our best model was in epoch 7`

`model.load_state_dict(torch.load(best_model_path, map_location=device))`

`model.to(device)`

`model.train()`

**We saved this 🏆 BEST MODEL 🏆 and continued the training with a lower learning rate of 2e-05.**

For instance, in the example below we demonstrate the results per epoch after the training on 15 epochs with the learning rate 3e-05 and then continuing the training on the lower learning rate for 9 epochs.

| Epoch | Learning Rate | Average Loss   | Validation Accuracy |
|-------|---------------|----------------|---------------------|
| 1/9   | 2e-05         | 0.533422770320 | 0.5625              |
| 2/9   | 2e-05         | 0.269488133614 | 0.5614583333333333  |
| 3/9   | 2e-05         | 0.143803662904 | 0.5666666666666667  |
| 4/9   | 2e-05         | 0.137765339206 | 0.5739583333333333  |
| 5/9   | 2e-05         | 0.076707698969 | 0.56875             |
| 6/9   | 2e-05         | 0.065273187822 | **0.5979166666666667**  |
| 7/9   | 2e-05         | 0.045589126971 | 0.596875            |
| 8/9   | 2e-05         | 0.199297960250 | 0.5739583333333333  |
| 9/9   | 2e-05         | 0.160073978136 | 0.584375            |

The best model from this training had the result on Kaggle of 0.600.

Our predictions made with the model 0.600 for Kaggle:

| Difficulty Level | Count |
|------------------|-------|
| A1               | 176   |
| A2               | 182   |
| B1               | 244   |
| B2               | 200   |
| C1               | 242   |
| C2               | 156   |

![predictions.png](https://github.com/AnyaLang/SBB_ML/blob/a61cf5434af67b37f57bf0cd083882fffb8aaa4a/all_predictions.png)

**Approach we took for the best model on Kaggle:**

After achieving initial results over 15 epochs with the a learning rate of 3e-05, we changed the learning rate to 1e-05 and continued the training for 3 more epochs. We saw the improvement of the model, so decided to proceed the training with slightly higher learning rate.**

We extended the training by an additional 6 epochs with a learning rate of 2e-05, which further refined our model.

| Epoch | Learning Rate | Average Loss        | Validation Accuracy | Notes                                          |
|-------|---------------|---------------------|---------------------|------------------------------------------------|
| 1/6   | 2e-05         | 0.06369797926811316 | 0.5697916666666667  | Saved as `best_model_lr2e-05_ep1_acc0.57.pt`    |
| 2/6   | 2e-05         | 0.0697462880416424  | 0.5864583333333333  | Saved as `best_model_lr2e-05_ep2_acc0.59.pt`    |
| 3/6   | 2e-05         | 0.08821526710380567 | 0.5697916666666667  |                                                |
| 4/6   | 2e-05         | 0.03653059935331839 | 0.584375            |                                                |
| 5/6   | 2e-05         | 0.03376048295150819 | 0.5864583333333333  |                                                |
| 6/6   | 2e-05         | 0.02625617888628161 | 0.5916666666666667  | Saved as `best_model_lr2e-05_ep6_acc0.59.pt`    |

Observing continuous improvement, we decided that maintaining the learning rate of 2e-05 was optimal and proceeded to extend the training for 3 more epochs, however, given one issue in the code, the training extended to additional **9 epochs**. Throughout this extended training period, we noticed that while the **average loss consistently decreased, the accuracy improvements on our model plateaued, showing only marginal gains**.

| Epoch | Learning Rate | Average Loss         | Validation Accuracy   |
|-------|---------------|----------------------|-----------------------|
| 1/9   | 2e-05         | 0.049511629976404944 | 0.5791666666666667    |
| 2/9   | 2e-05         | 0.17178194310969655  | 0.5864583333333333    |
| 3/9   | 2e-05         | 0.03391529844190397  | 0.5927083333333333    |
| 4/9   | 2e-05         | 0.01702627820344181  | 0.5989583333333334    |
| 5/9   | 2e-05         | 0.049664503030241273 | 0.6020833333333333    |
| 6/9   | 2e-05         | 0.028027213982947313 | 0.59375               |
| 7/9   | 2e-05         | 0.01858836026416005  | 0.5947916666666667    |
| 8/9   | 2e-05         | 0.015126325636394237 | 0.59375               |
| 9/9   | 2e-05         | 0.03083539136728177  | 0.5927083333333333    |


The decrease in average loss indicates that our model was becoming better at fitting the training data, effectively minimizing the error between the predicted values and the actual values. This can be attributed to the model's increasing proficiency in identifying patterns and reducing prediction errors for the specific scenarios presented during training. However, the minimal gains in accuracy suggest that these improvements in loss did not translate to a broader generalization capability on unseen data. 

While our best model achieved an accuracy of 0.610 on Kaggle by following this approach, we did not fully rerun the model to generate the evaluation metrics of the final results due to computational limitations and financial constraints, as our team had to purchase computational units via Google Colab Pro. However, following the approach of initially training the model for 15 epochs with a learning rate of 3e-05, and then continuing the training with a starting with a lower learning rate and adjusting it given the model results, should produce similar results.

| Description                     | Details                                        |
|---------------------------------|------------------------------------------------|
| **Initial Training**            | Learning rate of 1e-05                         |
| **Extended Fine-Tuning**        | Increased learning rate to 2e-05               |
| **Total Epochs**                | 15 (started with 6 and then 9)                 |
| **Epoch of Notable Outcome**    | 5 (from training with additional epochs)       |
| **Average Loss at Epoch 5**     | 0.049664503030241273                           |
| **Validation Accuracy at Epoch 5** | 0.6020833333333333                          |
| **Model Saved As**              | `best_model_lr2e-05_ep5_acc0.60.pt`            |
| **Final Loss**                  | 0.03083539136728177                            |
| **Final Accuracy**              | 0.6020833333333333                             |

Our predictions made with the model 0.610 for Kaggle:

| Difficulty Level | Count |
|------------------|-------|
| A1               | 180   |
| A2               | 242   |
| B1               | 180   |
| B2               | 242   |
| C1               | 173   |
| C2               | 183   |

![predictions_best.png](https://github.com/AnyaLang/SBB_ML/blob/4f95efcdcfd33810b7f1419ee02da33ddcb365f1/best_graph.png)

### 8️⃣ 
### 🖇️ Collective Overview
## Best Model: FlauBERT Model
## Ranking
## Streamlit Application
## Video
## Contributions
