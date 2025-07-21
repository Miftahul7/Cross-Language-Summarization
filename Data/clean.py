import json
import re
import unicodedata
import nltk
#nltk.download('punkt') 
from nltk.tokenize import sent_tokenize

def preprocess_text(text):
    """Clean, normalize, deduplicate sentences and optionally n-grams."""
    if not isinstance(text, str):
        return text

    # Normalize Unicode characters
    text = unicodedata.normalize("NFKC", text)

    # Remove URLs, emails, and .com links
    text = re.sub(r'https?://\S+|www\.\S+|\S+\.com\S*', '', text)
    text = re.sub(r'\S+@\S+', '', text)

    # Remove copyright-style footers
    text = re.sub(r"©?Copyright.*?\d{4}", "", text, flags=re.IGNORECASE)

    # Remove MediaWiki artifacts and placeholders
    text = re.sub(r'thumb\|[^|]+\|', '', text)
    text = re.sub(r'\[\[|\]\]', '', text)
    text = re.sub(r'\{\{.*?\}\}', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Fix repeated punctuation (e.g., "!!" -> "!")
    text = re.sub(r'([!?.,])\1+', r'\1', text)

    # Deduplicate exact sentences
    sentences = sent_tokenize(text)
    seen_sentences = set()
    unique_sentences = []
    for sent in sentences:
        sent = sent.strip()
        if sent not in seen_sentences:
            unique_sentences.append(sent)
            seen_sentences.add(sent)

    deduped_text = ' '.join(unique_sentences)

    #  filter repeated bigram phrases
    words = deduped_text.split()
    n = 2
    seen_ngrams = set()
    final_words = []
    for i in range(len(words)):
        ngram = ' '.join(words[i:i+n])
        if len(words[i:i+n]) == n and ngram not in seen_ngrams:
            final_words.append(words[i])
            seen_ngrams.add(ngram)
        elif len(words[i:i+n]) == 1:
            final_words.append(words[i])

    return ' '.join(final_words)

    
def preprocess_json_file(input_file, output_file):
    cleaned_data = []

    with open(input_file, "r", encoding="utf-8") as infile:
        data = [json.loads(line) for line in infile if line.strip()]

        for article in data:
            article['title'] = preprocess_text(article.get('title', ''))

            for section in article.get('sections', []):
                section['title'] = preprocess_text(section.get('title', ''))
                section['content'] = preprocess_text(section.get('content', ''))

                cleaned_refs = []
                for ref in section.get('references', []):
                    if ".com" in ref:
                        continue  # Remove references with ".com"
                    cleaned_refs.append(preprocess_text(ref))  # Clean 
                section['references'] = cleaned_refs

            cleaned_data.append(article)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for item in cleaned_data:
            json.dump(item, outfile, ensure_ascii=False)
            outfile.write('\n')

    print(f"Preprocessing complete. Output saved to: {output_file}")

preprocess_json_file("Data/Curated-20/bn/bn_books.json", "Data/Curated-20/bn/cleaned/bn_books.json")
