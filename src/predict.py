from pathlib import Path

import joblib

from auto_reply import get_reply
from preprocessing import clean_text   # SAME cleaning that was used for training

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model" / "ticket_classifier.joblib"

# If the top probability is below this, flag the ticket for a human to check
LOW_CONFIDENCE = 0.40


def load_model(path: Path = MODEL_PATH):
    """Load the trained pipeline (TF-IDF + Logistic Regression) from disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run 'python src/train.py' first."
        )
    return joblib.load(path)


def predict_ticket(model, text: str) -> dict:
    """Predict the category of ONE ticket description."""
    cleaned = clean_text(text)                          # 1. clean like in training
    probabilities = model.predict_proba([cleaned])[0]   # 2. probability for every category
    best = probabilities.argmax()                       # 3. position of the highest probability
    category = model.classes_[best]
    confidence = float(probabilities[best])

    return {
        "category": category,
        "confidence": confidence,
        "low_confidence": confidence < LOW_CONFIDENCE,
        "all_scores": dict(zip(model.classes_, probabilities.round(3))),
        "reply": get_reply(category),
    }


def main():
    model = load_model()
    print("=" * 55)
    print(" Support Ticket Classifier  (type 'quit' to exit)")
    print("=" * 55)

    while True:
        text = input("\nEnter ticket description: ").strip()
        if text.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        if not text:
            print("Please type something.")
            continue

        result = predict_ticket(model, text)
        print(f"\nPredicted Category: {result['category']}")
        print(f"Confidence:         {result['confidence'] * 100:.1f}%")
        if result["low_confidence"]:
            print("Note: low confidence - a human agent should check this ticket.")
        print(f"Suggested Reply:    {result['reply']}")


if __name__ == "__main__":
    main()