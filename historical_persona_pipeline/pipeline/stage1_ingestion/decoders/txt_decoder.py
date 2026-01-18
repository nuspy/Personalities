from pathlib import Path
from typing import Dict, Any, List

class TXTDecoder:
    def can_decode(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == '.txt'
    
    def decode(self, file_path: Path) -> dict:
        errors = []
        full_text = ""
        sections = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                full_text = f.read()
            
            # Simple segmentation by double newline
            raw_sections = full_text.split('\n\n')
            for i, content in enumerate(raw_sections):
                if content.strip():
                    sections.append({'type': 'paragraph', 'number': i + 1, 'content': content.strip()})
                    
        except UnicodeDecodeError:
            try:
                # Fallback to latin-1
                with open(file_path, 'r', encoding='latin-1') as f:
                    full_text = f.read()
                sections.append({'type': 'full_text', 'content': full_text})
            except Exception as e:
                errors.append(f"Encoding error: {str(e)}")
        except Exception as e:
            errors.append(f"Error: {str(e)}")
            
        return {
            'text': full_text,
            'metadata': {},
            'sections': sections,
            'errors': errors
        }

