import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

def add_translation_to_csv(babylonian, english, csv_file='babylonian_english.csv'):
    new_data = pd.DataFrame({'babylonian': [babylonian], 'english': [english]})
    
    try:
        existing_data = pd.read_csv(csv_file)
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
    except FileNotFoundError:
        updated_data = new_data
    
    updated_data.to_csv(csv_file, index=False)
    print(f"Added translation to {csv_file}")

def clean_babylonian_text(text):
    cleaned_text = re.sub(r"^\d+[\.']?\s*", '', text).strip()
    return trim_prefix_if_needed(cleaned_text)

def trim_prefix_if_needed(text):
    if text.startswith(". "):
        return text[2:]
    return text

def scrape_cdli_and_add_to_csv(base_url, pages=1, csv_file='babylonian_english.csv'):
    for page in range(1, pages + 1):
        url = f"{base_url}&page={page}"
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to retrieve page {page}")
            continue
        
        soup = BeautifulSoup(response.content, 'html.parser')
        search_cards = soup.find_all('div', class_='search-card-content')

        for card in search_cards:
            paragraphs = card.find_all('p', class_='mt-3')
            if not paragraphs:
                continue

            last_paragraph = paragraphs[-1]
            lines = last_paragraph.find_all('i')
            babylonian_text = None

            for line in lines:
                babylonian_text = line.find_previous_sibling(text=True).strip()
                if babylonian_text:
                    babylonian_text = clean_babylonian_text(babylonian_text)
                english_text = line.text.strip().replace('en:', '').strip()
                
                if babylonian_text and english_text:
                    add_translation_to_csv(babylonian_text, english_text, csv_file)
                    babylonian_text = None

# Example usage
base_url = 'https://cdli.mpiwg-berlin.mpg.de/search?f%5Blanguage%5D%5B%5D=Sumerian&f%5Batf_translation%5D%5B%5D=With&f%5Batf_translation_language%5D%5B%5D=en'
scrape_cdli_and_add_to_csv(base_url, pages=147, csv_file='babylonian_english.csv')
