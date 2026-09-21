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
        'text/markdown': FileFormat.MARKDOWN,
        'text/rtf': FileFormat.RTF,
        'application/rtf': FileFormat.RTF,
        'application/vnd.oasis.opendocument.text': FileFormat.ODT,
        'text/xml': FileFormat.XML_TEI,
        'application/xml': FileFormat.XML_TEI,
    }
    
    EXT_MAP = {
        '.txt': FileFormat.TXT,
        '.text': FileFormat.TXT,
        '.log': FileFormat.TXT,
        '.md': FileFormat.MARKDOWN,
        '.markdown': FileFormat.MARKDOWN,
        '.mdown': FileFormat.MARKDOWN,
        '.pdf': FileFormat.PDF,
        '.doc': FileFormat.DOC,
        '.docx': FileFormat.DOCX,
        '.rtf': FileFormat.RTF,
        '.odt': FileFormat.ODT,
        '.epub': FileFormat.EPUB,
        '.html': FileFormat.HTML,
        '.htm': FileFormat.HTML,
        '.xhtml': FileFormat.HTML,
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
            # Il riconoscimento per estensione copre tutti i casi d'uso reali:
            # python-magic aggiunge solo la capacita' di riconoscere un file
            # con estensione errata. Non e' un problema da segnalare a ogni
            # avvio, quindi resta a livello di debug.
            logger.debug(
                f"python-magic non disponibile ({e}): "
                "riconoscimento dei formati in base all'estensione."
            )
            self.use_magic = False
    
    # Formati contenitore ZIP: il MIME generico non li distingue,
    # l'estensione si'. Per questi l'estensione ha la precedenza.
    ZIP_BASED = {'.docx', '.epub', '.odt'}

    def detect(self, file_path: Path) -> Tuple[FileFormat, float]:
        ext = file_path.suffix.lower()
        if ext in self.ZIP_BASED and ext in self.EXT_MAP:
            return self.EXT_MAP[ext], 0.9

        # Try MIME detection first if available
        if self.use_magic:
            try:
                mime_type = self.magic.from_file(str(file_path))
                if mime_type in self.MIME_MAP:
                    return self.MIME_MAP[mime_type], 0.95
            except Exception as e:
                logger.debug(f"Magic detection failed for {file_path}: {e}")
        
        # Fallback to extension
        if ext in self.EXT_MAP:
            return self.EXT_MAP[ext], 0.7
        
        return FileFormat.UNKNOWN, 0.0
