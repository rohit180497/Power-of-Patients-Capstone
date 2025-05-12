import os
import json
import re

def clean_article_content(content):
    """
    Clean article content by removing timestamp lines, photo references, and other artifacts.
    """
    if not content:
        return ""

    # Split content into lines for easier processing
    lines = content.split('\n')
    cleaned_lines = []
    
    # Lines to skip (common patterns found in problematic content)
    skip_patterns = [
        r'Power of Patients$',  # Only when it appears as a standalone line
        r'\w{3} \d{1,2}, \d{4} \d{1,2}:\d{2}:\d{2} [AP]M',  # Date patterns like "Dec 21, 2022 9:58:24 AM"
        r'Photo Sourced from Here\.',
        r'Learn how to care for your physical and mental health after a TBI\.',
        r'Consider checking out our free app for more help and support today\.',
        r'Learn more below\.',
        r'Did you find this guide helpful',
        r'Keep reading\.'
    ]
    
    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue
            
        # Skip lines matching patterns
        should_skip = False
        for pattern in skip_patterns:
            if re.search(pattern, line):
                should_skip = True
                break
                
        if should_skip:
            continue
            
        # Add line to cleaned content
        cleaned_lines.append(line)
    
    # Join lines back into content
    cleaned_content = '\n\n'.join(cleaned_lines)
    
    # Remove redundant newlines
    cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content)
    
    return cleaned_content.strip()

def process_json_files(directory, output_directory=None):
    """
    Process all JSON files in the given directory and clean their content.
    Remove the 'sections' field and keep only the main fields.
    """
    if output_directory is None:
        output_directory = directory
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    print(f"Processing JSON files in {directory}")
    
    # Find all JSON files
    json_files = []
    for filename in os.listdir(directory):
        if filename.endswith('.json') and (filename.startswith('article_') or filename == 'all_articles.json'):
            json_files.append(os.path.join(directory, filename))
    
    print(f"Found {len(json_files)} JSON files")
    
    # Process each file
    for json_file in json_files:
        base_name = os.path.basename(json_file)
        print(f"Processing {base_name}")
        
        try:
            # Load JSON data
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Process a single article
            if isinstance(data, dict):
                # Clean content
                content = data.get('content', '')
                cleaned_content = clean_article_content(content)
                
                # Create simplified article with only the fields we want to keep
                simplified_article = {
                    'url': data.get('url', ''),
                    'title': data.get('title', ''),
                    'date': data.get('date', ''),
                    'author': data.get('author', ''),
                    'read_time': data.get('read_time', ''),
                    'content': cleaned_content
                }
                
                # Save cleaned data
                output_file = os.path.join(output_directory, f"simplified_{base_name}")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(simplified_article, f, indent=2)
                
                print(f"Saved simplified data to simplified_{base_name}")
            
            # Process all_articles.json
            elif isinstance(data, list):
                simplified_articles = []
                
                for article in data:
                    # Clean content
                    content = article.get('content', '')
                    cleaned_content = clean_article_content(content)
                    
                    # Create simplified article
                    simplified_article = {
                        'url': article.get('url', ''),
                        'title': article.get('title', ''),
                        'date': article.get('date', ''),
                        'author': article.get('author', ''),
                        'read_time': article.get('read_time', ''),
                        'content': cleaned_content
                    }
                    
                    simplified_articles.append(simplified_article)
                
                # Save cleaned data
                output_file = os.path.join(output_directory, f"simplified_{base_name}")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(simplified_articles, f, indent=2)
                
                print(f"Saved {len(simplified_articles)} simplified articles to simplified_{base_name}")
            
        except Exception as e:
            print(f"Error processing {base_name}: {e}")
    
    print("Processing complete")

def process_all_articles(articles_directory, save_directory=None):
    """
    Process individual article files and then create a simplified all_articles.json
    with sequential 4-digit IDs starting from 1001.
    """
    if save_directory is None:
        save_directory = articles_directory
    
    # Find all simplified article files
    all_articles = []
    
    for filename in os.listdir(articles_directory):
        if filename.startswith('simplified_article_') and filename.endswith('.json'):
            try:
                with open(os.path.join(articles_directory, filename), 'r', encoding='utf-8') as f:
                    article = json.load(f)
                    all_articles.append(article)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    # Sort articles by title for consistent ordering
    all_articles.sort(key=lambda x: x.get('title', ''))
    
    # Add sequential IDs
    start_id = 1001
    for i, article in enumerate(all_articles):
        # Generate a 4-digit ID
        article_id = str(start_id + i)
        
        # Add ID as the first field in the article
        article_with_id = {
            'id': article_id,
            'url': article.get('url', ''),
            'title': article.get('title', ''),
            'date': article.get('date', ''),
            'author': article.get('author', ''),
            'read_time': article.get('read_time', ''),
            'content': article.get('content', '')
        }
        
        # Replace the article with the version including ID
        all_articles[i] = article_with_id
    
    # Create output directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)
    
    # Save to all_articles.json
    output_file = os.path.join(save_directory, 'simplified_all_articles.json')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_articles, f, indent=2)
    
    print(f"Created simplified_all_articles.json with {len(all_articles)} articles")
    print(f"Added sequential IDs from {start_id} to {start_id + len(all_articles) - 1}")
    
    return all_articles

if __name__ == "__main__":
    # Set the directory path to your articles
    articles_directory = "power_of_patients_data/articles"
    
    # Uncomment and modify this line if your directory is different
    # articles_directory = "D:/workspace/git_projects/Power_Capstone/power_of_patients_data"
    
    # Process all individual article files
    # process_json_files(articles_directory)
    
    # Create a combined all_articles.json from the individual simplified files
    processed_directory = "power_of_patients_data/processed"
    save_directory = "power_of_patients_data"
    process_all_articles(processed_directory, save_directory)