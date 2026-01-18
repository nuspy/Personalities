from docx import Document
from pathlib import Path

class DOCXDecoder:
    def can_decode(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == '.docx'
    
    def decode(self, file_path: Path) -> dict:
        errors, sections, full_text, metadata = [], [], [], {}
        
        try:
            doc = Document(str(file_path))
            
            if doc.core_properties:
                metadata = {
                    'title': doc.core_properties.title or '',
                    'author': doc.core_properties.author or '',
                }
            
            current_section = {'content': [], 'heading': None}
            
            for para in doc.paragraphs:
                text = para.text.strip()
                if not text:
                    continue
                
                if para.style.name.startswith('Heading'):
                    if current_section['content']:
                        sections.append({
                            'heading': current_section['heading'],
                            'content': '\n'.join(current_section['content'])
                        })
                    current_section = {'content': [], 'heading': text}
                else:
                    current_section['content'].append(text)
                    full_text.append(text)
            
            if current_section['content']:
                sections.append({
                    'heading': current_section['heading'],
                    'content': '\n'.join(current_section['content'])
                })
                
        except Exception as e:
            errors.append(f"Error: {str(e)}")
        
        return {
            'text': "\n\n".join(full_text),
            'metadata': metadata,
            'sections': sections,
            'errors': errors
        }

