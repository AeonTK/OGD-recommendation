import requests
import json
import time
import urllib.parse
import sys
import logging

# --- Configuration ---
SPARQL_ENDPOINT = "https://data.europa.eu/sparql"
OUTPUT_FILE = "all_datasets.jsonl"
PAGE_SIZE = 10000  # We found 10k works, but you can lower it if you get errors
OVERLAP_SIZE = 100   # Your suggestion: how many records to re-fetch
POLITE_DELAY_SECONDS = 1  # Seconds to wait between requests

# --- Setup Logging ---
# Log to both standard output (console) and a file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("download.log")
    ]
)
# --- End Logging Setup ---

# The base SPARQL query we built
# We select the dataset URI, its title, description, and keywords
# We filter for English language text to avoid getting 24+ copies of each
BASE_QUERY = """
PREFIX dcat: <http://www.w3.org/ns/dcat#>
PREFIX odp:  <http://data.europa.eu/euodp/ontologies/ec-odp#>
PREFIX dct: <http://purl.org/dc/terms/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

SELECT ?dataset ?title ?description ?keywords
WHERE {
  # Main pattern: Get datasets, titles, and descriptions
  ?dataset a dcat:Dataset .
  ?dataset dct:title ?title .
  ?dataset dct:description ?description .
  
  FILTER(LANG(?title) = "en")
  FILTER(LANG(?description) = "en")

  # OPTIONAL Subquery: Get the aggregated keywords for each dataset
  OPTIONAL {
    SELECT ?dataset (GROUP_CONCAT(DISTINCT ?kw; SEPARATOR="; ") AS ?keywords)
    WHERE {
      ?dataset dcat:keyword ?kw .
    }
    GROUP BY ?dataset
  }
}
"""
# --- End of Configuration ---

def fetch_page(limit, offset):
    """
    Fetches a single page of results from the SPARQL endpoint.
    """
    logging.info(f"Fetching page: LIMIT={limit} OFFSET={offset}")
    
    # Build the full query with pagination
    paginated_query = BASE_QUERY + f" LIMIT {limit} OFFSET {offset}"
    
    # URL-encode the query for the POST body
    encoded_query = urllib.parse.urlencode({"query": paginated_query})
    
    headers = {
        "Accept": "application/sparql-results+json",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    
    try:
        response = requests.post(
            SPARQL_ENDPOINT, 
            data=encoded_query, 
            headers=headers, 
            timeout=60 # Set a generous timeout (in seconds)
        )
        
        # Check for HTTP errors
        response.raise_for_status() 
        
        # If we get here, the request was successful (e.g., 200 OK)
        return response.json()
        
    except requests.exceptions.HTTPError as e:
        logging.error(f"   HTTP Error: {e.response.status_code} {e.response.reason}")
    except requests.exceptions.Timeout:
        logging.warning("   Request timed out. The server is too busy.")
    except requests.exceptions.RequestException as e:
        logging.warning(f"   A request error occurred: {e}")
    except json.JSONDecodeError as e:
        logging.error(f"   Failed to parse JSON response: {e}")
        # Try to log the response text if possible, truncated
        if response and response.text:
             logging.error(f"   Response text (first 200 chars): {response.text[:200]}...")
        
    # If any error occurred, return None
    return None

def main():
    offset = 0
    total_fetched = 0
    
    # We open the file in 'append' mode ('a')
    # This lets us add to the end of the file without deleting its contents
    with open(OUTPUT_FILE, "a", encoding="utf-8") as outfile:
        while True:
            data = fetch_page(PAGE_SIZE, offset)
            
            if data is None:
                logging.warning("   Fetch failed. Retrying this page in 10 seconds...")
                time.sleep(10)
                continue # Try the same offset again

            results = data.get('results', {}).get('bindings', [])
            
            if not results:
                # If the 'bindings' array is empty, we've reached the end
                logging.info("No more results. Download complete.")
                break
            
            num_results = len(results)
            total_fetched += num_results
            logging.info(f"   Got {num_results} results. Total so far: {total_fetched:,}")

            # Write each result as a new line in the .jsonl file
            for record in results:
                # Simplify the record from SPARQL's format
                simplified_record = {
                    "dataset": record.get('dataset', {}).get('value'),
                    "title": record.get('title', {}).get('value'),
                    "description": record.get('description', {}).get('value'),
                    "keywords": record.get('keywords', {}).get('value') # Added this line
                }
                
                # Dump the dictionary as a JSON string and write it as one line
                outfile.write(json.dumps(simplified_record) + "\n")
            
            # Check if this was the last page
            if num_results < PAGE_SIZE:
                logging.info("Got fewer results than page size. Assuming this is the last page.")
                logging.info("Download complete.")
                break

            # Increment the offset for the next loop
            # We use an overlap (e.g., 10000 - 100 = 9900) for extra safety.
            offset += (PAGE_SIZE - OVERLAP_SIZE)
            
            # Be polite to the server
            logging.info(f"   Waiting {POLITE_DELAY_SECONDS} second(s)...")
            time.sleep(POLITE_DELAY_SECONDS)

    logging.info(f"All data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

