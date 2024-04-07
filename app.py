from enum import Enum
import streamlit as st

# from summarizer import Summarizer
from summarizer.sbert import SBertSummarizer

# import torch
from transformers import AutoModelWithLMHead, AutoTokenizer
import spacy
import clip
import pandas as pd
import numpy as np
import torch

device = 'cpu'
nlp = spacy.load("en_core_web_sm") # Load the English model
image_search_model = clip.load("ViT-B/32")
photo_ids = pd.read_csv("photo_ids.csv")
photo_ids = list(photo_ids['photo_id'])
photo_features = np.load("features.npy")
photo_features = torch.from_numpy(photo_features).float().to(device)

class SummaryType(Enum):
    EXTRACTIVE = 1
    ABSTRACTIVE = 2

def encode_search_query(search_query):
  with torch.no_grad():
    # Encode and normalize the search query using CLIP
    text_encoded = image_search_model.encode_text(clip.tokenize(search_query).to(device))
    text_encoded /= text_encoded.norm(dim=-1, keepdim=True)

  # Retrieve the feature vector
  return text_encoded

def find_best_matches(text_features, photo_features, photo_ids, results_count=3):
  # Compute the similarity between the search query and each photo using the Cosine similarity
  similarities = (photo_features @ text_features.T).squeeze(1)

  # Sort the photos by their similarity score
  best_photo_idx = (-similarities).argsort()

  # Return the photo IDs of the best matches
  return [photo_ids[i] for i in best_photo_idx[:results_count]]

def display_photo(photo_id):
  # Get the URL of the photo resized to have a width of 320px
  photo_image_url = f"https://unsplash.com/photos/{photo_id}/download?w=320"

  # Display the photo
  st.image(url=photo_image_url, width=500)

  # Display the attribution text
  st.write(f'Photo on <a target="_blank" href="https://unsplash.com/photos/{photo_id}">Unsplash</a> ')
  
def search_unslash(search_query, photo_features, photo_ids, results_count=3):
  # Encode the search query
  text_features = encode_search_query(search_query)

  # Find the best matches
  best_photo_ids = find_best_matches(text_features, photo_features, photo_ids, results_count)

  # Display the best photos
  for photo_id in best_photo_ids:
    display_photo(photo_id)

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
