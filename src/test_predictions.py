"""
test_predictions.py  (Task 5 - test with NEW tickets)
-----------------------------------------------------
Tests the model on tickets that are NOT in the dataset and prints a table:
    Test Input | Expected | Predicted | Confidence | Result

Run:  python src/test_predictions.py
Replace / add sentences in TEST_CASES with your own ones.
"""

from pathlib import Path

import pandas as pd

from predict import load_model, predict_ticket
from preprocessing import clean_text

ROOT = Path(__file__).resolve().parent.parent

# (ticket text, the category YOU expect)
TEST_CASES = [
    ("I can't remember my password and the site won't let me in", "Login Issue"),
    ("My account is blocked and I cannot enter the system", "Login Issue"),
    ("The app throws an error every time I press the save button", "Application Error"),
    ("Something went wrong and the page crashed with an error code", "Application Error"),
    ("Please send me the sales numbers for the last quarter", "Report"),
    ("Can you prepare a summary report of customer complaints for audit", "Report"),
    ("I want to change my phone number on my profile", "Account Update"),
    ("Kindly modify the email address registered with my account", "Account Update"),
    ("The website is loading really slowly this afternoon", "Performance"),
    ("Everything on the platform is lagging and takes forever to respond", "Performance"),
]

if __name__ == "__main__":
    pd.set_option("display.max_colwidth", 60)
    pd.set_option("display.width", 200)

    # Safety check: test sentences must NOT already be in the dataset
    dataset = pd.read_csv(ROOT / "data" / "tickets_clean.csv")
    known = set(dataset["clean_description"])
    for text, _ in TEST_CASES:
        if clean_text(text) in known:
            print(f"WARNING: this test sentence is already in the dataset -> {text}")

    model = load_model()
    rows = []
    for text, expected in TEST_CASES:
        result = predict_ticket(model, text)
        rows.append({
            "Test Input": text,
            "Expected": expected,
            "Predicted": result["category"],
            "Confidence": f"{result['confidence'] * 100:.0f}%",
            "Result": "Correct" if result["category"] == expected else "WRONG",
        })

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    correct = (table["Result"] == "Correct").sum()
    print(f"\nCorrect: {correct} / {len(table)}  ({correct / len(table) * 100:.0f}%)")