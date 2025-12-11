import json
import sys
import logging
import re
import html

# --- Configuration ---
INPUT_FILE = "all_datasets_api_async.jsonl"
OUTPUT_FILE = "all_datasets_UNIQUE.jsonl"
# --- End of Configuration ---

# --- Setup Logging ---
# Log to both standard output (console) and a file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("deduplication.log") # Separate log file
    ]
)
# --- End Logging Setup ---


def clean_text(text: str) -> str:
    """Clean title/description-style text.

    - Decode HTML entities (e.g. &agrave; -> à)
    - Strip HTML tags while keeping inner text
    - Simplify Markdown links/bold/italics
    - Remove simple ASCII decorations
    - Normalize whitespace
    """
    if not text:
        return ""

    # Decode HTML entities
    text = html.unescape(text)

    # Remove HTML tags (e.g. <a href="...">text</a> -> text)
    text = re.sub(r"<[^>]+>", "", text)

    # Markdown links: [Text](url) -> Text
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)

    # Markdown bold/italics: **Text** / __Text__ / *Text* / _Text_ -> Text
    text = re.sub(r"[\*_]{2,}(.*?)[\*_]{2,}", r"\1", text)
    text = re.sub(r"[\*_](.*?)[\*_]", r"\1", text)

    # ASCII decorations like =====
    text = re.sub(r"={3,}", "", text)

    # Normalize whitespace and newlines
    text = text.replace("\r\n", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_keywords(keywords_value):
    """Clean keywords field.

    - Treat "N_A" or empty as no keywords
    - Split on semicolons
    - Strip whitespace and drop empties
    Returns a list of keyword strings.
    """
    if not keywords_value:
        return []

    # Many records use a single string; tolerate lists as well.
    if isinstance(keywords_value, list):
        raw = ";".join(str(k) for k in keywords_value)
    else:
        raw = str(keywords_value)

    raw = raw.strip()
    if not raw or raw.upper() == "N_A":
        return []

    parts = [p.strip() for p in raw.split(";")]
    return [p for p in parts if p]

def main():
    logging.info("Starting deduplication...")
    logging.info(f"Input file: {INPUT_FILE}")
    logging.info(f"Output file: {OUTPUT_FILE}")

    # This set will store all the dataset URIs we've seen.
    # It will grow in RAM to hold all unique URIs.
    seen_dataset_uris = set()

    line_count = 0
    dupe_count = 0
    
    try:
        # Open the giant input file for reading
        with open(INPUT_FILE, 'r', encoding="utf-8") as infile:
            # Open the new, clean output file for writing
            with open(OUTPUT_FILE, 'w', encoding="utf-8") as outfile:
                
                # Read the input file one line at a time
                # This is very memory-efficient
                for line in infile:
                    line_count += 1
                    
                    # Give a progress update every 100,000 lines
                    if line_count % 100000 == 0:
                        logging.info(f"   Processed {line_count:,} lines...")

                    try:
                        # Turn the line of text back into a Python dictionary
                        record = json.loads(line)
                        
                        # Get the unique ID for this dataset
                        # We use .get() as a safety check in case the record is malformed
                        uri = record.get('dataset')
                        
                        if not uri:
                            logging.warning(f"   Skipping line {line_count} (missing 'dataset' URI)")
                            continue
                        
                        # Check if we have seen this URI before
                        if uri not in seen_dataset_uris:
                            # This is a new, unique dataset
                            seen_dataset_uris.add(uri)

                            # Clean fields before writing
                            cleaned_record = {
                                "dataset": uri,
                                "title": clean_text(record.get("title")),
                                "description": clean_text(record.get("description")),
                                "keywords": clean_keywords(record.get("keywords")),
                            }

                            outfile.write(json.dumps(cleaned_record, ensure_ascii=False) + "\n")
                        else:
                            # We've seen this URI. It's a duplicate.
                            dupe_count += 1

                    except json.JSONDecodeError:
                        # Handle cases where a line might be broken
                        logging.warning(f"   Skipping a bad JSON line at line {line_count}")

    except FileNotFoundError:
        logging.error(f"Error: Input file not found: {INPUT_FILE}")
        logging.error("Please run the download_datasets.py script first.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        sys.exit(1)

    logging.info("\nDeduplication complete!")
    logging.info("---")
    logging.info(f"Total lines read:      {line_count:,}")
    logging.info(f"Duplicates found:    {dupe_count:,}")
    logging.info(f"Unique datasets saved: {len(seen_dataset_uris):,}")
    logging.info("---")
    logging.info(f"Your clean, unique file is ready: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

