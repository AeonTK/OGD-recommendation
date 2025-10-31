import json
import sys
import logging

# --- Configuration ---
INPUT_FILE = "all_datasets.jsonl"
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
                            # 1. Add its URI to our set
                            seen_dataset_uris.add(uri)
                            # 2. Write the original, full line to our new file
                            outfile.write(line)
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

