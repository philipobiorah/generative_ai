from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Instantiate the model and tokenizer
pt_model = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(pt_model)
tokenizer = AutoTokenizer.from_pretrained(pt_model)

def get_prediction(review):
    """Given a review, return the predicted sentiment"""

    # Tokenize the review
    # (Get the response as tensors and not as a list)
    inputs = tokenizer(review, return_tensors="pt")

    # Perform the prediction (get the logits)
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted class (corresponding to the highest logit)
    predictions = torch.argmax(outputs.logits, dim=-1)

    return "positive" if predictions.item() == 1 else "negative"

# Example usage
if __name__ == "__main__":
    text = "I love using transformers library!"
    prediction = get_prediction(text)
    print(f"Predicted sentiment: {prediction}")

    # Check predictions
    review = "This movie is not so great :("
    print(f"Review: {review}")
    print(f"Sentiment: {get_prediction(review)}")
    assert get_prediction(review) == "negative", "The prediction should be negative"

    review = "This movie rocks!"
    print(f"Review: {review}")
    print(f"Sentiment: {get_prediction(review)}")
    assert get_prediction(review) == "positive", "The prediction should be positive"