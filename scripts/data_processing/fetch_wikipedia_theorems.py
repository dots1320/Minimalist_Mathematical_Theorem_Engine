import urllib.request
import urllib.parse
import json
import time
import os

def fetch_members(category, is_category=False):
    """Fetches titles of pages or subcategories inside a Wikipedia category."""
    titles = []
    cmtype = "subcat" if is_category else "page"
    url = f"https://en.wikipedia.org/w/api.php?action=query&list=categorymembers&cmtitle=Category:{category}&cmlimit=500&cmtype={cmtype}&format=json"
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'MathBot/1.0 (test@example.com)'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read())
            if 'query' in data and 'categorymembers' in data['query']:
                for member in data['query']['categorymembers']:
                    titles.append(member['title'])
    except Exception as e:
        print(f"Error fetching {category}: {e}")
        
    return titles

def fetch_extract(title):
    """Fetches the introductory paragraph (extract) of a Wikipedia page."""
    encoded_title = urllib.parse.quote(title)
    url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro=1&explaintext=1&titles={encoded_title}&format=json"
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'MathBot/1.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read())
            pages = data.get('query', {}).get('pages', {})
            for page_id, page_data in pages.items():
                if 'extract' in page_data:
                    extract = page_data['extract'].strip()
                    # Clean up extract by taking just the first paragraph generally
                    if extract:
                        first_para = extract.split('\n')[0]
                        # Remove text inside parentheses which usually contains pronunciations or alternate names
                        # Simple regex could work, but we'll leave it as is for broad knowledge
                        return first_para
    except Exception as e:
        print(f"Error fetching extract for {title}: {e}")
    
    return None

def main():
    root_category = "Mathematical_theorems"
    print(f"Fetching subcategories of {root_category}...")
    
    # Get direct subcategories (e.g., Category:Algebra_theorems, Category:Calculus_theorems)
    subcat_titles = fetch_members(root_category, is_category=True)
    print(f"Found {len(subcat_titles)} subcategories.")
    
    # Clean category names (remove "Category:" prefix)
    subcategories = [c.replace("Category:", "") for c in subcat_titles]
    subcategories.append(root_category) # Include the root category itself
    
    all_pages = set()
    for cat in subcategories:
        print(f"Fetching pages in {cat}...")
        pages = fetch_members(cat.replace(' ', '_'), is_category=False)
        all_pages.update(pages)
        time.sleep(0.5) # Be nice to API
        
    print(f"\nTotal unique theorem pages identified: {len(all_pages)}")
    
    theorems_data = []
    print("Fetching abstracts (this will take a while)...")
    
    for i, title in enumerate(list(all_pages)):
        if "theorem" not in title.lower() and "lemma" not in title.lower() and "inequality" not in title.lower() and "principle" not in title.lower() and "rule" not in title.lower():
            # Rough filter to make sure it's likely a theorem statement
            continue 
            
        print(f"[{i+1}/{len(all_pages)}] Fetching: {title}", end='\r')
        extract = fetch_extract(title)
        
        if extract and len(extract) > 40: # Ignore stubs
            theorems_data.append({
                "name": title,
                "statement": extract
            })
            
        time.sleep(0.2) # Rate limiting
        
        # for quick testing we can break early, but let's get everything
        # if i > 50: break
        
    print(f"\nSuccessfully extracted {len(theorems_data)} rigorous theorem statements.")
    
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data", "wikipedia_theorems.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(theorems_data, f, indent=4, ensure_ascii=False)
        
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
