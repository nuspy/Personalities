from bs4 import BeautifulSoup
from pathlib import Path

class HTMLDecoder:
    def can_decode(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in ['.html', '.htm']
    
    def decode(self, file_path: Path) -> dict:
        errors, sections, full_text, metadata = [], [], [], {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract basic metadata
            if soup.title:
                metadata['title'] = soup.title.string
            
            # Remove scripts and styles
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Simple cleaning of lines
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text_content = '\n'.join(chunk for chunk in chunks if chunk)
            
            full_text.append(text_content)
            sections.append({'type': 'body', 'content': text_content})

        except Exception as e:
            errors.append(f"Error: {str(e)}")
        
        return {
            'text': "\n\n".join(full_text),
            'metadata': metadata,
            'sections': sections,
            'errors': errors
        }

