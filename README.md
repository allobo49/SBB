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
### 3️⃣ Random Forest Classifier
### 4️⃣ Decision Tree Classifier
### 5️⃣ KNN Classifier
### 6️⃣ Neural Networks
### 7️⃣ FlauBERT Model
### 8️⃣ 
### 🖇️ Collective Overview
## Best Model: FlauBERT Model
## Ranking
## Streamlit Application
## Video
## Contributions
