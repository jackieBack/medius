from enum import Enum
import streamlit as st

# from summarizer import Summarizer
from summarizer.sbert import SBertSummarizer

# import torch
from transformers import AutoModelWithLMHead, AutoTokenizer
import spacy
import clip

nlp = spacy.load("en_core_web_sm") # Load the English model
image_search_model = clip.load("ViT-B/32")

class SummaryType(Enum):
    EXTRACTIVE = 1
    ABSTRACTIVE = 2

def paragraph_to_array(text):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences[:-1]

def get_summary_for_text(text, summary_option):
    if summary_option is SummaryType.EXTRACTIVE:
        model_extractive_summary = SBertSummarizer("paraphrase-MiniLM-L6-v2")
        result = model_extractive_summary(text, num_sentences=6)
    else:
        tokenizer = AutoTokenizer.from_pretrained("T5-base")
        model_abstractive_summary = AutoModelWithLMHead.from_pretrained(
            "T5-base", return_dict=True
        )

        inputs = tokenizer.encode(
            "sumarize: " + text, return_tensors="pt", truncation=True
        )
        output = model_abstractive_summary.generate(inputs, min_length=150, max_length=250)
        result = tokenizer.decode(output[0])
    return paragraph_to_array(result)


def get_images_for_text(text):
    """
    Placeholder function to get images for the given text.
    Replace this function with your actual logic to generate or fetch images.
    """
    import time

    time.sleep(2)

    # Example: Return a list of image URLs based on the input text
    # This is just a placeholder. Replace it with your own logic.
    return ["https://example.com/image1.jpg", "https://example.com/image2.jpg"]


def main():
    st.title("Text to Image Showcase")

    # Text input from the user
    user_input = st.text_area("Paste your text here:", height=150)
    # Radio buttons for summary type selection
    summary_type = st.radio(
        "Choose summary type:", ("Abstractive Summary", "Extractive Summary")
    )
    if summary_type == "Abstractive Summary":
        summary_type_enum = SummaryType.ABSTRACTIVE
    else:
        summary_type_enum = SummaryType.EXTRACTIVE

    if user_input:

        with st.spinner("Generating images... Please wait."):
            # image_urls = get_images_for_text(user_input)
            summaries = get_summary_for_text(user_input, summary_type_enum)
            for summary in summaries:
                st.write(summary)

        st.write("### Associated Images")
        # for url in image_urls:
        #     st.image(url, caption="Generated Image")


if __name__ == "__main__":
    main()
