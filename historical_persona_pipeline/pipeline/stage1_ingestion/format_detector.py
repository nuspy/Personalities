import logging
from pathlib import Path
from typing import Tuple
from ..data_models import FileFormat

logger = logging.getLogger(__name__)

class FormatDetector:
    MIME_MAP = {
        'text/plain': FileFormat.TXT,
        'application/pdf': FileFormat.PDF,
        'application/msword': FileFormat.DOC,
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': FileFormat.DOCX,
        'application/epub+zip': FileFormat.EPUB,
        'text/html': FileFormat.HTML,
        'text/xml': FileFormat.XML_TEI,
        'application/xml': FileFormat.XML_TEI,
    }
    
    EXT_MAP = {
        '.txt': FileFormat.TXT, 
        '.pdf': FileFormat.PDF,
        '.doc': FileFormat.DOC, 
        '.docx': FileFormat.DOCX,
        '.epub': FileFormat.EPUB, 
        '.html': FileFormat.HTML, 
        '.htm': FileFormat.HTML,
        '.xml': FileFormat.XML_TEI, 
        '.tei': FileFormat.XML_TEI,
    }
    
    def __init__(self):
        self.magic = None
        self.use_magic = False
        try:
            import magic
            # Attempt to initialize magic. 
            self.magic = magic.Magic(mime=True)
            self.use_magic = True
        except Exception as e:
            logger.warning(f"Could not initialize python-magic: {e}. Falling back to extension-based detection.")
            self.use_magic = False
    
    def detect(self, file_path: Path) -> Tuple[FileFormat, float]:
        # Try MIME detection first if available
        if self.use_magic:
            try:
                mime_type = self.magic.from_file(str(file_path))
                if mime_type in self.MIME_MAP:
                    return self.MIME_MAP[mime_type], 0.95
            except Exception as e:
                logger.debug(f"Magic detection failed for {file_path}: {e}")
        
        # Fallback to extension
        ext = file_path.suffix.lower()
        if ext in self.EXT_MAP:
            return self.EXT_MAP[ext], 0.7
        
        return FileFormat.UNKNOWN, 0.0
