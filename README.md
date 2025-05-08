# NLP-Sentiment-Classifier

# Age Bias in AI/ML
**CS 8321 Lab 1**  
**Authors:** Abdul Wasay, Aitor Elfau, Sam Yao

This project investigates age-related bias in machine learning models, particularly in sentiment analysis and word embeddings. Inspired by class materials adapted from Dr. Robyn Speer and Dr. Larson, our work explores how age-associated language influences ML predictions, potentially leading to discriminatory practices in real-world applications like hiring.

---

## 🧠 Project Objective

We explore whether sentiment classifiers and word embeddings reflect **age bias**, especially in modern vs. older language usage. This bias could be embedded in models used in AI-driven hiring pipelines or recommendation systems.

- **Why it matters:**  
  Automated systems using ML might rate resumes or communications differently based solely on language style, which can correlate with age. This leads to **unintentional discrimination**, reinforcing societal ageism.

- **Example Scenario:**  
  A Gen Z phrase like _"rizzed up 150 customers"_ may be rated positively, while a more traditionally phrased resume line referencing the 1970s may be penalized simply due to the model's exposure to newer language.

---

## 🔍 Research Questions

1. Do modern word embeddings like **GloVe** or **ConceptNet Numberbatch** contain age-related biases?
2. How do sentiment scores differ between slang from older generations vs. Gen Z?
3. Does the **sentiment lexicon** (Hu and Liu, 2004) favor modern language?
4. What is the **impact of embedding source and classifier** on sentiment evaluation of age-associated terms?

### Hypothesis:
> Word embeddings trained on contemporary corpora will exhibit **bias toward modern slang**, potentially assigning more positive sentiment to newer terms and penalizing older ones.

---

## ⚙️ Technologies & Libraries Used

- **Python** (Jupyter Notebook)
- **Pandas**, **NumPy**
- **Scikit-learn** (SGDClassifier, train_test_split, accuracy_score)
- **Matplotlib**, **Seaborn** (data visualization)
- **NLTK** (tokenization, preprocessing)
- **GloVe 840B 300D Embeddings**
- **Hu & Liu Sentiment Lexicon (2004)**

---

## 🧪 Methodology

### 1. Load Pre-trained Word Embeddings
- GloVe embeddings (`glove.840B.300d.txt`) were used for robust vector representations.
- Embedding matrix loaded and indexed with over 2M words.

### 2. Load Sentiment Lexicon
- Positive and negative words from Hu & Liu lexicon used to label data.
- Words not found in GloVe were filtered out.

### 3. Train a Sentiment Classifier
- Used **SGDClassifier** with `log_loss` to predict sentiment (1 for positive, -1 for negative).
- Achieved ~93.6% accuracy on test split.

### 4. Score Custom Phrases
- Function `text_to_sentiment()` processes input phrases and computes an **average log-odds sentiment** based on tokenized words.

---

## 🧪 Bias Evaluation: Old vs. New Slang

Two dictionaries were created to evaluate sentiment:
- `pos_terms`: positive slang from **older** and **newer** generations.
- `neg_terms`: negative slang from **older** and **newer** generations.

Each term was passed through the classifier, and results were compared using **mean sentiment scores** and **visualized via boxplots**.

### Sample Output Table
| Word      | Sentiment Score | Group     |
|-----------|------------------|-----------|
| groovy    | 3.2              | old_pos   |
| slaps     | -5.0             | new_pos   |
| dweeb     | -8.3             | old_neg   |
| sus       | -2.1             | new_neg   |

### Statistical Analysis
- **Paired t-test** used to compare sentiment scores between old and new terms.
- This test is appropriate for paired comparisons (same context, different vocabularies).

---

## 📈 Results

- Sentiment models often rated **new slang more positively** than older terms, even when the older terms originally had positive connotations.
- Some neutral Gen Z words (e.g., _slaps_, _rizz_) were given **negative sentiment**, likely due to lack of presence in older lexicons.
- Indicates **vocabulary bias in embedding + lexicon models**.

---

## ⚠️ Limitations

- Lexicons like Hu & Liu are dated and may not include modern slang.
- GloVe embeddings may underrepresent rare or modern informal terms.
- Cultural context and polysemy of words (e.g., "killer") can distort sentiment.

---

## 📚 References

- Hu, M., & Liu, B. (2004). Mining and summarizing customer reviews.
- Speer, R. (2017). How to Make a Racist AI Without Really Trying.  
- [ConceptNet Blog](http://blog.conceptnet.io/posts/2017/how-to-make-a-racist-ai-without-really-trying/)
- GloVe: [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)
- https://nypost.com, https://parade.com, https://yourdictionary.com, etc. (for slang terms)

---

## ✅ Conclusion

The results show a clear **bias in sentiment analysis** tools based on word embeddings and lexicons. As AI systems are increasingly used in domains like hiring, care must be taken to **audit and mitigate bias**, especially **age-related discrimination** that can emerge from outdated or narrow training data.

---
