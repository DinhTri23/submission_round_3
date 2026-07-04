import os
import re
from markdownify import markdownify as md

def slugify(title: str) -> str:
    """Creates a filesystem-safe filename."""
    s = title.lower().strip()
    s = re.sub(r'[^\w\s-]', '', s)
    s = re.sub(r'[\s_-]+', '-', s)
    return s

def save_to_markdown(article: dict, output_dir: str):
    """Converts HTML to Markdown and saves to file."""
    os.makedirs(output_dir, exist_ok=True)
    
    slug = slugify(article['title'])
    filename = f"{slug}.md"
    filepath = os.path.join(output_dir, filename)
    
    # Transform HTML to Markdown
    content = md(article['body'] or "")
    
    # Add metadata header (YAML front matter)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"---\ntitle: {article['title']}\nurl: {article['html_url']}\n---\n\n")
        f.write(content)
        
    return filepath