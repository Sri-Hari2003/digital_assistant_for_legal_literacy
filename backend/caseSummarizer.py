from transformers import pipeline

# Load the model once globally
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


def chunk_text(text, max_tokens=900):
    """Split long text into smaller chunks under token limit."""
    paragraphs = text.split("\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) < max_tokens:
            current_chunk += para + "\n"
        else:
            chunks.append(current_chunk)
            current_chunk = para + "\n"

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def summarize_text(text: str) -> str:
    # Truncate to model's max token limit (approx 1024 tokens ≈ 3500 characters)
    max_chars = 3500
    text = text[:max_chars]

    try:
        summary = summarizer(text, max_length=180, min_length=30, do_sample=False)
        return summary[0]["summary_text"]
    except Exception as e:
        return f"Summarization failed: {str(e)}"