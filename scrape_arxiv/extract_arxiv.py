import json
from bs4 import BeautifulSoup
import re

def clean_text(text):
    # Remove extra whitespace and newlines
    return re.sub(r'\s+', ' ', text).strip()

def extract_arxiv_info(html_file):
    with open(html_file, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

    results = []
    # Find all arxiv-result elements
    for article in soup.find_all('li', class_='arxiv-result'):
        article_info = {}
        
        # Extract PDF link
        pdf_link = article.find('a', href=lambda x: x and 'pdf' in x)
        if pdf_link:
            article_info['pdf_link'] = pdf_link['href']
        
        # Extract title
        title = article.find('p', class_='title')
        if title:
            # Clean up the title text by removing extra spaces and newlines
            article_info['title'] = clean_text(title.get_text())
        
        # Extract abstract
        abstract = article.find('span', class_='abstract-full')
        if abstract:
            article_info['abstract'] = clean_text(abstract.get_text())
        else:
            # If full abstract not found, try getting the short version
            abstract_short = article.find('span', class_='abstract-short')
            if abstract_short:
                article_info['abstract'] = clean_text(abstract_short.get_text())
        
        # Extract arXiv ID
        arxiv_link = article.find('a', href=lambda x: x and 'arxiv.org/abs/' in x)
        if arxiv_link:
            article_info['arxiv_id'] = arxiv_link.get_text()
        
        # Only add articles that have at least some information
        if article_info:
            results.append(article_info)

    # Save to JSON file
    with open('arxiv_extracted_data.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Successfully extracted {len(results)} articles to arxiv_extracted_data.json")

# Run the extraction
extract_arxiv_info('arxiv_results.html')