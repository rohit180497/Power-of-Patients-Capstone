#!/usr/bin/env python3
"""
CDC Traumatic Brain Injury Content Scraper & Organizer
=====================================================
A comprehensive tool to extract, organize, and catalog all CDC TBI resources
including publications, reports, guidelines, and educational materials.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import os
import re
from urllib.parse import urljoin, urlparse
from pathlib import Path
import time
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple
import pdfkit
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class CDCTBIScraper:
    def __init__(self, base_output_dir: str = "CDC_TBI_Resources"):
        """Initialize the CDC TBI scraper with output directory structure."""
        self.base_url = "https://www.cdc.gov"
        self.base_output_dir = Path(base_output_dir)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('cdc_scraper.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Create directory structure
        self.setup_directories()
        
        # Initialize data storage
        self.all_resources = []
        self.failed_downloads = []
        
        # Setup Chrome driver for HTML to PDF conversion
        self.setup_webdriver()

    def setup_directories(self):
        """Create organized directory structure for different content types."""
        directories = [
            "01_Research_Publications/MMWR_Articles",
            "01_Research_Publications/Journal_Articles",
            "01_Research_Publications/Surveillance_Reports", 
            "02_Clinical_Guidelines/Healthcare_Providers",
            "02_Clinical_Guidelines/Pediatric_Guidelines",
            "03_Educational_Materials/HEADS_UP_Resources",
            "03_Educational_Materials/Fact_Sheets",
            "03_Educational_Materials/Training_Materials",
            "04_Policy_Reports/Congress_Reports",
            "04_Policy_Reports/Government_Reports",
            "05_Data_Statistics/Surveillance_Data",
            "05_Data_Statistics/Trend_Reports",
            "06_Prevention_Resources/Sports_Safety",
            "06_Prevention_Resources/General_Prevention",
            "07_Recovery_Resources/Rehabilitation_Guidelines",
            "07_Recovery_Resources/Patient_Resources",
            "metadata",
            "converted_pdfs",
            "original_html"
        ]
        
        for directory in directories:
            (self.base_output_dir / directory).mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Created directory structure in {self.base_output_dir}")

    def setup_webdriver(self):
        """Setup Chrome webdriver for HTML to PDF conversion."""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            self.driver = webdriver.Chrome(options=chrome_options)
            self.logger.info("Chrome webdriver initialized successfully")
        except Exception as e:
            self.logger.warning(f"Could not initialize webdriver: {e}")
            self.driver = None

    def extract_publication_metadata(self, soup: BeautifulSoup, url: str) -> Dict:
        """Extract comprehensive metadata from a publication page."""
        metadata = {
            'url': url,
            'title': '',
            'authors': [],
            'publish_date': '',
            'publication_type': '',
            'journal': '',
            'doi': '',
            'abstract': '',
            'keywords': [],
            'content_length': 0,
            'scraped_date': datetime.now().isoformat(),
            'pdf_available': False,
            'pdf_url': ''
        }
        
        # Extract title
        title_selectors = [
            'h1', 'title', '.article-title', '.publication-title',
            '.page-title', '[class*="title"]'
        ]
        for selector in title_selectors:
            title_elem = soup.select_one(selector)
            if title_elem and title_elem.get_text().strip():
                metadata['title'] = title_elem.get_text().strip()
                break
        
        # Extract authors
        author_patterns = [
            r'([A-Z][a-zA-Z\s]+(?:,\s*[A-Z][a-zA-Z\s]+)*)\.\s*\d{4}',
            r'By\s+([A-Z][a-zA-Z\s,]+)',
            r'Author[s]?:\s*([A-Z][a-zA-Z\s,]+)'
        ]
        
        text_content = soup.get_text()
        for pattern in author_patterns:
            matches = re.findall(pattern, text_content)
            if matches:
                authors = [name.strip() for name in matches[0].split(',')]
                metadata['authors'] = authors
                break
        
        # Extract publication date
        date_patterns = [
            r'(\d{4})',
            r'(\w+\s+\d{1,2},\s*\d{4})',
            r'(\d{1,2}/\d{1,2}/\d{4})'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text_content)
            if matches:
                metadata['publish_date'] = matches[0]
                break
        
        # Determine publication type based on URL and content
        if 'mmwr' in url.lower():
            metadata['publication_type'] = 'MMWR Article'
        elif 'surveillance' in url.lower():
            metadata['publication_type'] = 'Surveillance Report'
        elif 'congress' in url.lower():
            metadata['publication_type'] = 'Congressional Report'
        elif 'heads-up' in url.lower():
            metadata['publication_type'] = 'Educational Material'
        elif any(term in url.lower() for term in ['guideline', 'recommendation']):
            metadata['publication_type'] = 'Clinical Guideline'
        else:
            metadata['publication_type'] = 'Research Publication'
        
        # Look for PDF links
        pdf_links = soup.find_all('a', href=re.compile(r'\.pdf$', re.I))
        if pdf_links:
            metadata['pdf_available'] = True
            metadata['pdf_url'] = urljoin(url, pdf_links[0]['href'])
        
        # Extract DOI
        doi_pattern = r'10\.\d{4,}/[^\s]+'
        doi_matches = re.findall(doi_pattern, text_content)
        if doi_matches:
            metadata['doi'] = doi_matches[0]
        
        # Get content length
        metadata['content_length'] = len(text_content)
        
        return metadata

    def download_pdf(self, pdf_url: str, filename: str, category_dir: str) -> bool:
        """Download PDF file to appropriate directory."""
        try:
            response = self.session.get(pdf_url, stream=True)
            response.raise_for_status()
            
            filepath = self.base_output_dir / category_dir / filename
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            self.logger.info(f"Downloaded PDF: {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download PDF {pdf_url}: {e}")
            self.failed_downloads.append({'url': pdf_url, 'error': str(e)})
            return False

    def convert_html_to_pdf(self, url: str, filename: str, category_dir: str) -> bool:
        """Convert HTML page to PDF using webdriver."""
        if not self.driver:
            return False
            
        try:
            self.driver.get(url)
            time.sleep(3)  # Wait for page to load
            
            # Save HTML first
            html_path = self.base_output_dir / "original_html" / f"{filename}.html"
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(self.driver.page_source)
            
            # Convert to PDF using browser's print function
            pdf_path = self.base_output_dir / "converted_pdfs" / f"{filename}.pdf"
            
            # Use pdfkit as fallback
            try:
                pdfkit.from_url(url, str(pdf_path))
                self.logger.info(f"Converted to PDF: {filename}")
                return True
            except Exception as e:
                self.logger.warning(f"pdfkit failed for {url}: {e}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to convert {url} to PDF: {e}")
            return False

    def categorize_content(self, metadata: Dict) -> str:
        """Determine the appropriate category directory for content."""
        url = metadata['url'].lower()
        title = metadata['title'].lower()
        pub_type = metadata['publication_type'].lower()
        
        # MMWR Articles
        if 'mmwr' in url or 'mmwr' in pub_type:
            return "01_Research_Publications/MMWR_Articles"
        
        # Journal Articles
        if any(term in url for term in ['pubmed', 'doi.org', 'journal']):
            return "01_Research_Publications/Journal_Articles"
        
        # Surveillance Reports
        if 'surveillance' in url or 'surveillance' in title:
            return "01_Research_Publications/Surveillance_Reports"
        
        # Congressional Reports
        if 'congress' in url or 'congress' in title:
            return "04_Policy_Reports/Congress_Reports"
        
        # Clinical Guidelines
        if any(term in url for term in ['guideline', 'clinical', 'provider']):
            return "02_Clinical_Guidelines/Healthcare_Providers"
        
        # HEADS UP Resources
        if 'heads-up' in url or 'heads up' in title:
            return "03_Educational_Materials/HEADS_UP_Resources"
        
        # Fact Sheets
        if 'fact' in title or 'sheet' in title:
            return "03_Educational_Materials/Fact_Sheets"
        
        # Prevention Resources
        if any(term in title for term in ['prevention', 'safety', 'sports']):
            return "06_Prevention_Resources/Sports_Safety"
        
        # Default to research publications
        return "01_Research_Publications/Journal_Articles"

    def scrape_publications_page(self) -> List[Dict]:
        """Scrape the main publications page for all resource links."""
        publications_url = "https://www.cdc.gov/traumatic-brain-injury/publications/"
        
        try:
            response = self.session.get(publications_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            resources = []
            
            # Find all publication links
            publication_links = soup.find_all('a', href=True)
            
            for link in publication_links:
                href = link.get('href')
                if not href:
                    continue
                
                # Skip navigation and non-content links
                if any(skip in href for skip in ['#', 'javascript:', 'mailto:', 'tel:']):
                    continue
                
                # Convert relative URLs to absolute
                full_url = urljoin(self.base_url, href)
                
                # Filter for relevant content
                if any(domain in full_url for domain in [
                    'cdc.gov/mmwr', 'cdc.gov/traumatic-brain-injury',
                    'pubmed.ncbi.nlm.nih.gov', 'stacks.cdc.gov'
                ]):
                    resources.append({
                        'url': full_url,
                        'link_text': link.get_text().strip(),
                        'source_page': publications_url
                    })
            
            # Remove duplicates
            seen_urls = set()
            unique_resources = []
            for resource in resources:
                if resource['url'] not in seen_urls:
                    seen_urls.add(resource['url'])
                    unique_resources.append(resource)
            
            self.logger.info(f"Found {len(unique_resources)} unique resources")
            return unique_resources
            
        except Exception as e:
            self.logger.error(f"Failed to scrape publications page: {e}")
            return []

    def scrape_additional_sections(self) -> List[Dict]:
        """Scrape additional CDC TBI sections for more resources."""
        additional_urls = [
            "https://www.cdc.gov/traumatic-brain-injury/index.html",
            "https://www.cdc.gov/traumatic-brain-injury/data-research/facts-stats/index.html",
            "https://www.cdc.gov/traumatic-brain-injury/hcp/communication-resources/",
            "https://www.cdc.gov/heads-up/about/index.html"
        ]
        
        all_resources = []
        
        for url in additional_urls:
            try:
                response = self.session.get(url)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find all relevant links
                links = soup.find_all('a', href=True)
                for link in links:
                    href = link.get('href')
                    if href and not any(skip in href for skip in ['#', 'javascript:', 'mailto:']):
                        full_url = urljoin(self.base_url, href)
                        if 'cdc.gov' in full_url and any(term in full_url for term in [
                            'traumatic-brain-injury', 'heads-up', 'concussion', 'tbi'
                        ]):
                            all_resources.append({
                                'url': full_url,
                                'link_text': link.get_text().strip(),
                                'source_page': url
                            })
                
                time.sleep(1)  # Be respectful to the server
                
            except Exception as e:
                self.logger.error(f"Failed to scrape {url}: {e}")
        
        return all_resources

    def process_single_resource(self, resource: Dict) -> Dict:
        """Process a single resource - extract metadata and download content."""
        url = resource['url']
        self.logger.info(f"Processing: {url}")
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract metadata
            metadata = self.extract_publication_metadata(soup, url)
            metadata.update(resource)  # Add original resource info
            
            # Generate filename
            safe_title = re.sub(r'[^\w\s-]', '', metadata['title'][:100])
            safe_title = re.sub(r'\s+', '_', safe_title)
            filename = f"{safe_title}_{datetime.now().strftime('%Y%m%d')}"
            
            # Determine category
            category = self.categorize_content(metadata)
            metadata['category'] = category
            metadata['filename'] = filename
            
            # Download or convert content
            if metadata['pdf_available'] and metadata['pdf_url']:
                pdf_filename = f"{filename}.pdf"
                success = self.download_pdf(metadata['pdf_url'], pdf_filename, category)
                metadata['download_success'] = success
            else:
                # Convert HTML to PDF
                pdf_filename = f"{filename}.pdf"
                success = self.convert_html_to_pdf(url, filename, category)
                metadata['conversion_success'] = success
            
            time.sleep(1)  # Rate limiting
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to process {url}: {e}")
            return {**resource, 'error': str(e), 'processed': False}

    def run_full_scrape(self):
        """Run the complete scraping process."""
        self.logger.info("Starting CDC TBI content scraping...")
        
        # Get all resource URLs
        self.logger.info("Collecting resource URLs...")
        publication_resources = self.scrape_publications_page()
        additional_resources = self.scrape_additional_sections()
        
        # Combine and deduplicate
        all_resources = publication_resources + additional_resources
        seen_urls = set()
        unique_resources = []
        for resource in all_resources:
            if resource['url'] not in seen_urls:
                seen_urls.add(resource['url'])
                unique_resources.append(resource)
        
        self.logger.info(f"Processing {len(unique_resources)} unique resources...")
        
        # Process each resource
        processed_resources = []
        for i, resource in enumerate(unique_resources, 1):
            self.logger.info(f"Progress: {i}/{len(unique_resources)}")
            processed = self.process_single_resource(resource)
            processed_resources.append(processed)
            self.all_resources.append(processed)
        
        # Save comprehensive metadata
        self.save_metadata()
        
        # Generate summary report
        self.generate_summary_report()
        
        self.logger.info("Scraping completed!")

    def save_metadata(self):
        """Save all collected metadata to various formats."""
        metadata_dir = self.base_output_dir / "metadata"
        
        # Save as JSON
        json_path = metadata_dir / "all_resources_metadata.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.all_resources, f, indent=2, ensure_ascii=False)
        
        # Save as CSV
        csv_path = metadata_dir / "all_resources_metadata.csv"
        df = pd.DataFrame(self.all_resources)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        # Save failed downloads
        if self.failed_downloads:
            failed_path = metadata_dir / "failed_downloads.json"
            with open(failed_path, 'w') as f:
                json.dump(self.failed_downloads, f, indent=2)
        
        self.logger.info(f"Metadata saved to {metadata_dir}")

    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        report_path = self.base_output_dir / "CDC_TBI_Collection_Summary.md"
        
        # Calculate statistics
        total_resources = len(self.all_resources)
        by_category = {}
        by_type = {}
        successful_downloads = 0
        
        for resource in self.all_resources:
            category = resource.get('category', 'Unknown')
            pub_type = resource.get('publication_type', 'Unknown')
            
            by_category[category] = by_category.get(category, 0) + 1
            by_type[pub_type] = by_type.get(pub_type, 0) + 1
            
            if resource.get('download_success') or resource.get('conversion_success'):
                successful_downloads += 1
        
        # Write report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# CDC Traumatic Brain Injury Resource Collection\n\n")
            f.write(f"**Collection Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Total Resources:** {total_resources}\n")
            f.write(f"**Successfully Downloaded/Converted:** {successful_downloads}\n")
            f.write(f"**Success Rate:** {successful_downloads/total_resources*100:.1f}%\n\n")
            
            f.write("## Resources by Category\n\n")
            for category, count in sorted(by_category.items()):
                f.write(f"- **{category.replace('_', ' ').title()}:** {count} resources\n")
            
            f.write("\n## Resources by Publication Type\n\n")
            for pub_type, count in sorted(by_type.items()):
                f.write(f"- **{pub_type}:** {count} resources\n")
            
            f.write("\n## Directory Structure\n\n")
            f.write("```\n")
            f.write("CDC_TBI_Resources/\n")
            f.write("├── 01_Research_Publications/\n")
            f.write("│   ├── MMWR_Articles/\n")
            f.write("│   ├── Journal_Articles/\n")
            f.write("│   └── Surveillance_Reports/\n")
            f.write("├── 02_Clinical_Guidelines/\n")
            f.write("├── 03_Educational_Materials/\n")
            f.write("├── 04_Policy_Reports/\n")
            f.write("├── 05_Data_Statistics/\n")
            f.write("├── 06_Prevention_Resources/\n")
            f.write("├── 07_Recovery_Resources/\n")
            f.write("├── metadata/ (JSON & CSV files)\n")
            f.write("├── converted_pdfs/ (HTML→PDF conversions)\n")
            f.write("└── original_html/ (Source HTML files)\n")
            f.write("```\n\n")
            
            f.write("## Usage Instructions\n\n")
            f.write("1. **Browse by Category:** Navigate to numbered directories for organized content\n")
            f.write("2. **Search Metadata:** Use CSV/JSON files in metadata/ for searchable information\n")
            f.write("3. **Access Original Sources:** All PDFs include source URLs in metadata\n")
            f.write("4. **Quality Control:** Check failed_downloads.json for any missing content\n\n")
            
            f.write("## Metadata Fields\n\n")
            f.write("Each resource includes comprehensive metadata:\n")
            f.write("- Title, Authors, Publication Date\n")
            f.write("- Source URL, DOI (if available)\n")
            f.write("- Publication Type, Category\n")
            f.write("- Content Length, Keywords\n")
            f.write("- Download/Conversion Status\n")

    def cleanup(self):
        """Clean up resources."""
        if self.driver:
            self.driver.quit()

def main():
    """Main execution function."""
    scraper = CDCTBIScraper()
    
    try:
        scraper.run_full_scrape()
    except KeyboardInterrupt:
        scraper.logger.info("Scraping interrupted by user")
    except Exception as e:
        scraper.logger.error(f"Scraping failed: {e}")
    finally:
        scraper.cleanup()

if __name__ == "__main__":
    # Install required packages first
    print("Installing required packages...")
    # os.system("pip install requests beautifulsoup4 pandas selenium pdfkit")
    
    # Note: You'll also need to install Chrome/Chromium and ChromeDriver
    print("\nIMPORTANT: Make sure you have:")
    print("1. Chrome browser installed")
    print("2. ChromeDriver in your PATH")
    print("3. wkhtmltopdf installed (for pdfkit)")
    print("\nStarting scraper...")
    
    main()