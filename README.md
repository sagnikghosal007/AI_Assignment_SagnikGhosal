# AI-Based Support Ticket Classification

A small machine-learning project that reads the text of a customer support ticket
(for example *"I forgot my password"*) and predicts its **category**
(for example *Login Issue*). It also shows a **confidence score** and suggests a
ready-made reply.

Built for the **AI Fresher Technical Assignment**.

---

## Problem Statement

Support teams receive many tickets written in free text. Sorting them by hand is slow
and error-prone. This project automatically classifies each ticket into one of five
categories so it can be routed to the right team quickly.

| Input | Output |
|---|---|
| Ticket description (text) | Category, confidence %, suggested reply |

**Categories:** Login Issue, Application Error, Report, Account Update, Performance

---

## How It Works

```
Ticket text -> clean text -> TF-IDF (text to numbers) -> Logistic Regression -> category + confidence
```

1. **Preprocessing:** remove duplicates and missing values, lowercase the text, remove punctuation.
2. **Features:** TF-IDF (single words and two-word pairs, English stop words removed).
3. **Model:** Logistic Regression, chosen after comparing it with Naive Bayes using 5-fold cross-validation.
4. **Prediction:** the category with the highest probability; that probability is the confidence.

---

## Technologies Used

- Python 3
- pandas, numpy (data handling)
- matplotlib, seaborn (charts)
- scikit-learn (TF-IDF, models, metrics)
- joblib (saving the trained model)
- Streamlit (web interface)

---

## Project Structure

```
AI_Assignment_<YourName>/
|-- app.py                      # Streamlit web app
|-- requirements.txt
|-- README.md
|-- report.pdf                  # Project report
|-- data/
|   |-- tickets.csv             # raw dataset
|   `-- tickets_clean.csv       # cleaned dataset
|-- src/
|   |-- generate_dataset.py     # creates tickets.csv (run once)
|   |-- preprocessing.py        # Task 1: data cleaning
|   |-- eda.py                  # Task 2: analysis and charts
|   |-- train.py                # Tasks 3-4: train, test, evaluate, save model
|   |-- predict.py              # Task 5: predict a new ticket
|   |-- test_predictions.py     # tests the model on new tickets
|   `-- auto_reply.py           # Bonus: suggested replies
|-- model/
|   |-- ticket_classifier.joblib
|   `-- metrics.txt
`-- screenshots/                # charts, confusion matrix, output screenshots (dataset.png, prediction.png, ...)
```

---

## Dataset

- **Records:** 154 raw tickets, 149 after cleaning.
- **Fields:** `ticket_id`, `ticket_description`, `category`, `priority`, `status`.
- **Categories:** 5, with about 30 tickets each (balanced).
- **Source:** created for this assignment with AI assistance. Descriptions were written
  with varied wording per category, priority and status were assigned randomly, and a few
  duplicates and missing values were added on purpose to demonstrate data cleaning.

---

## Installation

1. Install **Python 3.9 or newer** from python.org (tick *"Add python.exe to PATH"*).
2. Clone the repository and open the folder:
   ```
   git clone <your-repository-url>
   cd <repository-folder>
   ```
3. Install the libraries:
   ```
   pip install -r requirements.txt
   ```

> **Windows tip:** if `pip` or `python` is "not recognized", use the launcher instead:
> `py -m pip install -r requirements.txt`, `py src/train.py`, `py -m streamlit run app.py`.

---

## How to Run

Run all commands from the **project root folder**.

**1. Clean the data** (Task 1)
```
python src/preprocessing.py
```

**2. Explore the data** (Task 2) - saves charts in `screenshots/`
```
python src/eda.py
```

**3. Train and evaluate the model** (Tasks 3-4) - saves the model in `model/`
```
python src/train.py
```

**4. Predict a ticket from the command line** (Task 5)
```
python src/predict.py
```
Type a ticket description when asked, or `quit` to exit.

**5. Test on new tickets**
```
python src/test_predictions.py
```

**6. Run the web app** (Task 6)
```
streamlit run app.py
```
A browser tab opens at `http://localhost:8501`.

> `src/generate_dataset.py` only needs to run once to create `data/tickets.csv`.
> The dataset is already included, and running it again would overwrite the file.

---

## Sample Input and Output

**Command line**

```
Enter ticket description: I forgot my password and cannot Login

Predicted Category: Login Issue
Confidence:         57.8%
Suggested Reply:    Hello, sorry you are having trouble signing in. Please try the
                    'Forgot Password' option on the login page to reset your password...
```

**Web app:** type a ticket, click **Predict**, and the app shows the predicted category,
the confidence, the suggested reply, and the score for every category.

---

## Results

Evaluated on a held-out test set of 30 tickets (80/20 stratified split, `random_state=42`).

| Metric | Score |
|---|---|
| Accuracy | 96.7% |
| Precision (weighted) | 97.1% |
| Recall (weighted) | 96.7% |
| F1 score (weighted) | 96.6% |

| Model | 5-fold CV accuracy (training data) |
|---|---|
| Naive Bayes | 88.3% |
| Logistic Regression (chosen) | 89.1% |

On 10 brand-new tickets that are not in the dataset, the model got **9 of 10** correct.
See `report.pdf` for the confusion matrix and a discussion of the one mistake.

---

## Limitations

- Small, consistently worded dataset (about 150 tickets), so real-world accuracy would be lower.
- The model only knows words seen in training; synonyms and new phrases can be misclassified.
- Confidence values are fairly low (about 36-64%) because of the small dataset; they show
  how sure the model is, not a guarantee of correctness.
- Each ticket gets one category, but real tickets can belong to several.

## Future Improvements

- Train on a larger, real dataset and tune hyperparameters.
- Try sentence embeddings or a fine-tuned transformer such as BERT.
- Support multi-label prediction, retrain regularly, and add a database and an API.
- Use an LLM to draft replies tailored to each ticket.

---

## Use of AI Tools

AI tools (Claude by Anthropic) were used to help explain concepts, draft the ticket
descriptions in the dataset, and help write and debug the code. All code was run, tested
and reviewed by the author, who can explain how each part works.

## Author

Sagnik Ghosal

## Coordinator

Inner Eye Consultancy Services 