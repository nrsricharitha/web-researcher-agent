import requests
from bs4 import BeautifulSoup
import html2text

def scrape_webpage(url: str, max_chars: int = 12000) -> str:
    """
    Downloads a webpage and extracts clean, readable text/markdown.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return f"Failed to retrieve page, HTTP Status: {response.status_code}"
            
        # Parse content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove noisy elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            element.decompose()
            
        # Convert to clean markdown-like text
        h = html2text.HTML2Text()
        h.ignore_links = True
        h.ignore_images = True
        h.ignore_emphasis = True
        h.body_width = 0  # No line wrapping
        text = h.handle(str(soup))
        
        # Clean extra whitespaces/newlines
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase for phrase in lines if phrase)
        cleaned_text = "\n".join(chunks)
        
        return cleaned_text[:max_chars]
        
    except Exception as e:
        return f"Error occurred while scraping {url}: {str(e)}"
