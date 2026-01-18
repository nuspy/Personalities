import fitz  # PyMuPDF
from pathlib import Path

class PDFDecoder:
    def can_decode(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == '.pdf'
    
    def decode(self, file_path: Path) -> dict:
        errors, sections, full_text, metadata = [], [], [], {}
        
        try:
            doc = fitz.open(str(file_path))
            metadata = {
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'page_count': len(doc),
            }
            
            for page_num, page in enumerate(doc):
                text = page.get_text("text")
                if text.strip():
                    full_text.append(text)
                    sections.append({'type': 'page', 'number': page_num + 1, 'content': text})
            
            doc.close()
        except Exception as e:
            errors.append(f"Error: {str(e)}")
        
        return {
            'text': "\n\n".join(full_text),
            'metadata': metadata,
            'sections': sections,
            'errors': errors
        }

