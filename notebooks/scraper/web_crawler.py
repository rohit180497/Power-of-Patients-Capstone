import time
import json
import os
import re
import requests
import random
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
from bs4 import BeautifulSoup

def setup_driver():
    """Setup Chrome webdriver with headless option."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    
    # Add user agent to avoid detection
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0"
    ]
    chrome_options.add_argument(f"--user-agent={random.choice(user_agents)}")
    
    driver = webdriver.Chrome(options=chrome_options)
    
    # Disable webdriver detection
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    return driver

def extract_direct_article_links(driver):
    """Extract article links directly from href attributes."""
    print("Extracting article links from href attributes...")
    
    # Get all links on the page
    links = driver.find_elements(By.TAG_NAME, 'a')
    
    # Filter for blog post links
    article_urls = set()
    for link in links:
        try:
            href = link.get_attribute('href')
            if href and '/blog/' in href and href.count('/') >= 3:
                # Only add actual article URLs, not category filters
                article_urls.add(href)
        except:
            continue
    
    print(f"Found {len(article_urls)} direct article URLs")
    return list(article_urls)

def simulate_realistic_scrolling(driver, max_scrolls=150, max_time=300):
    """Simulate realistic scrolling behavior to load all content."""
    print("Starting realistic scrolling simulation...")
    scroll_count = 0
    start_time = time.time()
    previous_height = driver.execute_script("return document.body.scrollHeight")
    previous_url_count = 0
    
    # Keep track of heights where we've already scrolled
    scrolled_positions = set()
    
    while scroll_count < max_scrolls and (time.time() - start_time) < max_time:
        # Random scroll amount between 300 and 800 pixels
        scroll_amount = random.randint(300, 800)
        current_position = driver.execute_script("return window.pageYOffset;")
        
        # Scroll down
        target_position = current_position + scroll_amount
        driver.execute_script(f"window.scrollTo(0, {target_position});")
        
        # Add random pause to simulate human behavior (1-3 seconds)
        time.sleep(random.uniform(1, 3))
        
        # Sometimes do multiple small scrolls
        if random.random() < 0.3:
            for _ in range(random.randint(2, 5)):
                small_scroll = random.randint(100, 300)
                driver.execute_script(f"window.scrollTo(0, {target_position + small_scroll});")
                target_position += small_scroll
                time.sleep(random.uniform(0.5, 1.5))
        
        # Record the position we've scrolled to
        scrolled_positions.add(target_position)
        
        # Check if we've scrolled to the bottom
        current_height = driver.execute_script("return document.body.scrollHeight")
        current_position = driver.execute_script("return window.pageYOffset;")
        viewport_height = driver.execute_script("return window.innerHeight;")
        
        # If we're near the bottom of the page
        if current_position + viewport_height >= current_height - 200:
            # Wait longer at the bottom to ensure everything loads
            print("Reached bottom of page, waiting for content to load...")
            time.sleep(random.uniform(5, 8))
            
            # Check if page height increased
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == current_height:
                # Try to force load more content by scrolling up and down
                print("Trying to force load more content...")
                driver.execute_script("window.scrollTo(0, 0);")
                time.sleep(2)
                driver.execute_script(f"window.scrollTo(0, {current_height});")
                time.sleep(5)
                
                newest_height = driver.execute_script("return document.body.scrollHeight")
                if newest_height == new_height:
                    # One last attempt - refresh the page and scroll to where we were
                    if random.random() < 0.3:  # Only do this sometimes
                        print("Refreshing page and continuing scroll...")
                        driver.refresh()
                        time.sleep(5)
                        driver.execute_script(f"window.scrollTo(0, {current_position});")
                        time.sleep(2)
                    else:
                        # Try one more extraction before concluding
                        direct_links = extract_direct_article_links(driver)
                        if len(direct_links) > previous_url_count:
                            previous_url_count = len(direct_links)
                        else:
                            print("No more new content found, ending scroll")
                            break
        
        # Check if we found new content
        if scroll_count % 5 == 0:
            direct_links = extract_direct_article_links(driver)
            print(f"Found {len(direct_links)} URLs after {scroll_count} scrolls")
            if len(direct_links) > previous_url_count:
                previous_url_count = len(direct_links)
            elif scroll_count > 30:  # Only start checking for no new content after 30 scrolls
                print("No new URLs found in the last 5 scrolls")
                # Try scrolling to a random new position
                unscrolled_height = current_height - max(scrolled_positions) if scrolled_positions else current_height
                if unscrolled_height > 500:
                    random_position = max(scrolled_positions) + random.randint(100, int(unscrolled_height/2))
                    print(f"Trying new scroll position: {random_position}")
                    driver.execute_script(f"window.scrollTo(0, {random_position});")
                    time.sleep(3)
        
        scroll_count += 1
        print(f"Scroll {scroll_count}/{max_scrolls}, total time: {int(time.time() - start_time)}s")
        
    print(f"Completed {scroll_count} scrolls in {int(time.time() - start_time)} seconds")
    return

def extract_links_from_html(html):
    """Extract blog post links directly from HTML."""
    soup = BeautifulSoup(html, 'html.parser')
    article_urls = set()
    
    # Find all links
    for a in soup.find_all('a', href=True):
        href = a['href']
        if '/blog/' in href and href.count('/') >= 3:
            # Ensure it's a full URL
            if not href.startswith('http'):
                href = f"https://www.powerofpatients.com{href}" if href.startswith('/') else f"https://www.powerofpatients.com/{href}"
            
            # Skip category filter URLs
            if href.endswith('/blog/') or '/category/' in href:
                continue
                
            article_urls.add(href)
    
    return list(article_urls)

def visit_article_for_metadata(url):
    """Visit an article page to extract metadata."""
    print(f"Visiting article for metadata: {url}")
    
    driver = setup_driver()
    driver.get(url)
    
    # Wait for article to load
    time.sleep(4)
    
    # Default metadata
    metadata = {
        'title': url.split('/')[-1].replace('-', ' ').title(),
        'url': url,
        'date': "Unknown",
        'read_time': "Unknown",
        'author': "Unknown"
    }
    
    try:
        # Extract title
        try:
            title_element = driver.find_element(By.CSS_SELECTOR, 'h1, h2.post-title, [data-hook="post-title"] h2')
            metadata['title'] = title_element.text.strip()
        except NoSuchElementException:
            print(f"Could not find title element for {url}")
            
        # Extract date - try multiple selectors
        try:
            # Try the most common selector
            date_element = driver.find_element(By.CSS_SELECTOR, '[data-hook="time-ago"]')
            metadata['date'] = date_element.get_attribute('title') or date_element.text.strip()
        except NoSuchElementException:
            try:
                # Try alternative selectors
                date_element = driver.find_element(By.CSS_SELECTOR, '.post-metadata__date, .post-date, .date')
                metadata['date'] = date_element.get_attribute('title') or date_element.text.strip()
            except NoSuchElementException:
                # Try to find it in the HTML using regex
                html = driver.page_source
                date_patterns = [
                    r'data-hook="time-ago"[^>]*title="([^"]+)"',
                    r'data-hook="time-ago"[^>]*>([^<]+)<',
                    r'class="[^"]*post-date[^"]*"[^>]*>([^<]+)<',
                    r'class="[^"]*date[^"]*"[^>]*>([^<]+)<'
                ]
                
                for pattern in date_patterns:
                    date_match = re.search(pattern, html)
                    if date_match:
                        metadata['date'] = date_match.group(1).strip()
                        break
        
        # Extract read time - try multiple selectors
        try:
            read_time_element = driver.find_element(By.CSS_SELECTOR, '[data-hook="time-to-read"]')
            metadata['read_time'] = read_time_element.get_attribute('title') or read_time_element.text.strip()
        except NoSuchElementException:
            try:
                read_time_element = driver.find_element(By.CSS_SELECTOR, '.post-metadata__readTime, .read-time')
                metadata['read_time'] = read_time_element.get_attribute('title') or read_time_element.text.strip()
            except NoSuchElementException:
                # Try to find it in the HTML using regex
                html = driver.page_source
                read_time_patterns = [
                    r'data-hook="time-to-read"[^>]*title="([^"]+)"',
                    r'data-hook="time-to-read"[^>]*>([^<]+)<',
                    r'class="[^"]*post-metadata__readTime[^"]*"[^>]*>([^<]+)<',
                    r'class="[^"]*read-time[^"]*"[^>]*>([^<]+)<',
                    r'(\d+)\s+min\s+read'
                ]
                
                for pattern in read_time_patterns:
                    read_time_match = re.search(pattern, html)
                    if read_time_match:
                        metadata['read_time'] = read_time_match.group(1).strip()
                        break
        
        # Extract author
        try:
            author_element = driver.find_element(By.CSS_SELECTOR, '[data-hook="user-name"]')
            metadata['author'] = author_element.text.strip()
        except NoSuchElementException:
            try:
                author_element = driver.find_element(By.CSS_SELECTOR, '.user-name, .author')
                metadata['author'] = author_element.text.strip()
            except NoSuchElementException:
                print(f"Could not find author element for {url}")
    
    except Exception as e:
        print(f"Error extracting metadata from {url}: {e}")
    
    finally:
        # Close the driver
        driver.quit()
    
    return metadata

def crawl_blog(output_dir="power_of_patients_data"):
    """
    Main function to crawl the Power of Patients blog.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup driver for blog page
    driver = setup_driver()
    blog_url = "https://www.powerofpatients.com/blog"
    
    print(f"Opening blog page: {blog_url}")
    driver.get(blog_url)
    
    # Wait for initial load
    time.sleep(5)
    
    # Use realistic scrolling to load all content
    simulate_realistic_scrolling(driver, max_scrolls=150, max_time=300)
    
    # Extract article URLs using multiple methods
    direct_urls = extract_direct_article_links(driver)
    html_urls = extract_links_from_html(driver.page_source)
    
    # Combine all URLs and remove duplicates
    all_urls = list(set(direct_urls + html_urls))
    print(f"Total unique article URLs found: {len(all_urls)}")
    
    # Close the main blog page browser
    driver.quit()
    
    # Process each article for metadata
    articles = []
    for i, url in enumerate(all_urls):
        print(f"Processing article {i+1}/{len(all_urls)}: {url}")
        
        metadata = visit_article_for_metadata(url)
        articles.append(metadata)
        
        # Save progress every 10 articles
        if (i+1) % 10 == 0 or i == len(all_urls) - 1:
            with open(os.path.join(output_dir, "article_list_progress.json"), 'w', encoding='utf-8') as f:
                json.dump(articles, f, indent=2)
            print(f"Saved progress ({i+1}/{len(all_urls)} articles)")
        
        # Add a delay between requests
        time.sleep(random.uniform(2, 4))
    
    # Save final list of articles
    with open(os.path.join(output_dir, "article_list.json"), 'w', encoding='utf-8') as f:
        json.dump(articles, f, indent=2)
    
    print(f"Found and processed {len(articles)} articles with metadata")
    print(f"All data saved to {os.path.join(output_dir, 'article_list.json')}")
    
    return articles

def extract_article_content(url):
    """
    Extract content from an individual article page.
    """
    print(f"Extracting content from: {url}")
    
    # Setup driver for individual article
    driver = setup_driver()
    driver.get(url)
    
    # Wait for article to load
    time.sleep(6)
    
    try:
        # Use a longer timeout for article content
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "article, .post-content, [data-hook='post-content']"))
        )
    except TimeoutException:
        print(f"Timeout waiting for article content at {url}")
    
    # Extract article content
    article_data = {
        "url": url,
        "title": "",
        "date": "",
        "author": "",
        "read_time": "",
        "content": "",
        "sections": []
    }
    
    try:
        # Extract title
        title_element = driver.find_element(By.CSS_SELECTOR, 'h1, h2.post-title, [data-hook="post-title"] h2')
        article_data["title"] = title_element.text.strip()
    except NoSuchElementException:
        print("Could not find title element")
        # Extract from URL as fallback
        article_data["title"] = url.split('/')[-1].replace('-', ' ').title()
    
    try:
        # Extract date
        date_element = driver.find_element(By.CSS_SELECTOR, '[data-hook="time-ago"]')
        article_data["date"] = date_element.get_attribute('title') or date_element.text.strip()
    except NoSuchElementException:
        print("Could not find date element")
    
    try:
        # Extract author
        author_element = driver.find_element(By.CSS_SELECTOR, '[data-hook="user-name"]')
        article_data["author"] = author_element.text.strip()
    except NoSuchElementException:
        print("Could not find author element")
    
    try:
        # Extract read time
        read_time_element = driver.find_element(By.CSS_SELECTOR, '[data-hook="time-to-read"]')
        article_data["read_time"] = read_time_element.get_attribute('title') or read_time_element.text.strip()
    except NoSuchElementException:
        print("Could not find read time element")
    
    # Extract article content using BeautifulSoup for more reliable parsing
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    # Find the main content container
    content_element = soup.select_one('article') or soup.select_one('[data-hook="post-content"]')
    
    if not content_element:
        # Try alternative selectors
        content_element = soup.select_one('main') or soup.select_one('.post-content') or soup.select_one('.blog-post')
    
    if content_element:
        # Extract all paragraphs
        paragraphs = content_element.find_all('p')
        content_text = "\n\n".join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        article_data["content"] = content_text
        
        # Extract sections with headings
        sections = []
        headings = content_element.find_all(['h2', 'h3', 'h4'])
        
        for heading in headings:
            section_content = []
            current = heading.next_sibling
            
            # Collect all content until the next heading
            while current and current.name not in ['h2', 'h3', 'h4']:
                if current.name == 'p':
                    section_content.append(current.get_text().strip())
                current = current.next_sibling
            
            sections.append({
                "heading": heading.get_text().strip(),
                "content": "\n\n".join(section_content)
            })
            
        article_data["sections"] = sections
    
    driver.quit()
    return article_data

def process_article_content(article_list_file, output_dir="power_of_patients_data"):
    """
    Process article content from a list of articles.
    """
    # Load article list
    with open(article_list_file, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    print(f"Processing content for {len(articles)} articles")
    
    # Process each article
    full_articles = []
    for i, article in enumerate(articles):
        print(f"Processing article {i+1}/{len(articles)}: {article['title']}")
        
        try:
            # Extract content from each article URL
            article_data = extract_article_content(article['url'])
            
            # Copy metadata from article list
            if article_data["date"] == "":
                article_data["date"] = article.get("date", "Unknown")
            if article_data["read_time"] == "":
                article_data["read_time"] = article.get("read_time", "Unknown")
            if article_data["author"] == "":
                article_data["author"] = article.get("author", "Unknown")
            
            full_articles.append(article_data)
            
            # Save individual article data
            article_filename = f"article_{i+1:03d}.json"
            
            with open(os.path.join(output_dir, article_filename), 'w', encoding='utf-8') as f:
                json.dump(article_data, f, indent=2)
            
            print(f"Saved article data to {os.path.join(output_dir, article_filename)}")
            
            # Save progress every 5 articles
            if (i+1) % 5 == 0 or i == len(articles) - 1:
                with open(os.path.join(output_dir, "full_articles_progress.json"), 'w', encoding='utf-8') as f:
                    json.dump(full_articles, f, indent=2)
                print(f"Saved progress ({i+1}/{len(articles)} articles)")
            
            # Add a delay between article processing
            time.sleep(random.uniform(2, 4))
            
        except Exception as e:
            print(f"Error processing article {article['url']}: {str(e)}")
    
    # Save all articles to a single file
    with open(os.path.join(output_dir, "all_articles.json"), 'w', encoding='utf-8') as f:
        json.dump(full_articles, f, indent=2)
    
    print(f"Processed {len(full_articles)} out of {len(articles)} articles")
    print(f"All data saved to {os.path.join(output_dir, 'all_articles.json')}")
    
    return full_articles

if __name__ == "__main__":
    # Step 1: Crawl blog to get article URLs with metadata
    # articles = crawl_blog()
    
    # Step 2: Process article content (optional - uncomment to run)
    article_list_file = "power_of_patients_data/article_list.json"
    process_article_content(article_list_file)