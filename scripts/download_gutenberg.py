#!/usr/bin/env python3
"""
Download top Project Gutenberg books for language model training
"""

import urllib.request
import urllib.error
import re
import time
from pathlib import Path

def get_top_books(num_books=200):
    """
    Get list of top downloaded books from Project Gutenberg
    Returns list of (book_id, title, author) tuples
    """
    print("Fetching top books list...")
    
    # Gutenberg's top 100 yesterday page - this gives us the most popular books
    url = "https://www.gutenberg.org/browse/scores/top"
    
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            html = response.read().decode('utf-8')
        
        # Parse the HTML to extract book IDs
        # Look for links like /ebooks/1342
        book_links = re.findall(r'/ebooks/(\d+)', html)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_books = []
        for book_id in book_links:
            if book_id not in seen:
                seen.add(book_id)
                unique_books.append(book_id)
        
        print(f"Found {len(unique_books)} books")
        return unique_books[:num_books]  # Top 200
        
    except Exception as e:
        print(f"Error fetching book list: {e}")
        # Fallback: use a manually curated list of very popular books
        return [
            "1342", "11", "74", "2701", "1080", "84", "1661", "98", "2600", "345",
            "76", "174", "215", "2554", "6130", "1260", "408", "2641", "205", "2591"
        ]

def clean_gutenberg_text(text):
    """
    Remove Project Gutenberg headers and footers
    """
    # Find start and end markers
    start_pattern = r'\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*'
    end_pattern = r'\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*'
    
    # Remove everything before the start marker
    start_match = re.search(start_pattern, text, re.IGNORECASE | re.DOTALL)
    if start_match:
        text = text[start_match.end():]
    
    # Remove everything after the end marker
    end_match = re.search(end_pattern, text, re.IGNORECASE | re.DOTALL)
    if end_match:
        text = text[:end_match.start()]
    
    # Clean up extra whitespace
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Multiple newlines to double
    text = text.strip()
    
    return text

def download_book(book_id, output_dir):
    """
    Download a single book by ID
    Returns (success, filename, size_kb)
    """
    
    # Try different text formats in order of preference
    urls_to_try = [
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",  # UTF-8
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",    # ASCII
        f"https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8"       # Alternative
    ]
    
    for url in urls_to_try:
        try:
            print(f"  Trying {url}")
            
            with urllib.request.urlopen(url, timeout=30) as response:
                if response.status == 200:
                    text = response.read().decode('utf-8')
                    
                    # Clean the text
                    cleaned_text = clean_gutenberg_text(text)
                    
                    if len(cleaned_text) < 1000:  # Skip very short texts
                        print(f"  Skipping {book_id}: too short ({len(cleaned_text)} chars)")
                        return False, None, 0
                    
                    # Save to file
                    filename = f"{book_id}.txt"
                    filepath = output_dir / filename
                    
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(cleaned_text)
                    
                    size_kb = len(cleaned_text) // 1024
                    print(f"  ✓ Downloaded {book_id}: {size_kb}KB")
                    return True, filename, size_kb
                
        except urllib.error.URLError as e:
            print(f"  Failed {url}: {e}")
            continue
        except Exception as e:
            print(f"  Failed {url}: {e}")
            continue
    
    print(f"  ✗ Could not download book {book_id}")
    return False, None, 0

def main():
    # Create output directory
    output_dir = Path("data/gutenberg_corpus")
    output_dir.mkdir(exist_ok=True)
    
    print("Project Gutenberg Corpus Builder")
    print("=" * 40)
    
    # Get list of top books
    book_ids = get_top_books(num_books=200)
    
    print(f"\nDownloading {len(book_ids)} books...")
    print("=" * 40)
    
    successful = 0
    total_size_kb = 0
    
    for i, book_id in enumerate(book_ids, 1):
        print(f"[{i}/{len(book_ids)}] Book {book_id}")
        
        success, filename, size_kb = download_book(book_id, output_dir)
        
        if success:
            successful += 1
            total_size_kb += size_kb
        
        # Be polite to Gutenberg's servers
        time.sleep(0.5)
        
        # Progress update every 20 books
        if i % 20 == 0:
            print(f"\nProgress: {successful}/{i} successful, {total_size_kb/1024:.1f}MB so far\n")
    
    print("\n" + "=" * 40)
    print("DOWNLOAD COMPLETE")
    print(f"Successfully downloaded: {successful}/{len(book_ids)} books")
    print(f"Total size: {total_size_kb/1024:.1f}MB")
    print(f"Average book size: {total_size_kb/successful:.0f}KB")
    print(f"Files saved to: {output_dir.absolute()}")
    
    # Create a simple catalog
    catalog_path = output_dir / "catalog.txt"
    with open(catalog_path, 'w') as f:
        f.write(f"Project Gutenberg Corpus\n")
        f.write(f"Books downloaded: {successful}\n")
        f.write(f"Total size: {total_size_kb/1024:.1f}MB\n")
        f.write(f"Book IDs: {', '.join(book_ids[:successful])}\n")
    
    print(f"Catalog saved to: {catalog_path}")

if __name__ == "__main__":
    main()