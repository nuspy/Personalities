from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime
import logging
import uuid

from ..stage_base import PipelineStage
from ..data_models import IngestionResult, TextSegment, FileFormat
from .format_detector import FormatDetector
from .language_detector import LanguageDetector
from .text_normalizer import TextNormalizer

# Import Decoders
from .decoders.txt_decoder import TXTDecoder
from .decoders.pdf_decoder import PDFDecoder
from .decoders.docx_decoder import DOCXDecoder
from .decoders.html_decoder import HTMLDecoder
# XML/TEI and EPUB decoders would be imported here once implemented

logger = logging.getLogger(__name__)

class IngestionStage(PipelineStage):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.format_detector = FormatDetector()
        self.language_detector = LanguageDetector()
        self.normalizer = TextNormalizer()
        
        # Initialize decoders
        self.decoders = [
            TXTDecoder(),
            PDFDecoder(),
            DOCXDecoder(),
            HTMLDecoder(),
            # Add others here
        ]
        
    def run(self, file_paths: List[Path]) -> IngestionResult:
        self.logger.info(f"Starting ingestion of {len(file_paths)} files")
        self.progress_update.emit(0, "Initializing ingestion...")
        
        segments: List[TextSegment] = []
        errors: List[Dict[str, Any]] = []
        language_counts: Dict[str, int] = {}
        
        total_files = len(file_paths)
        
        for idx, file_path in enumerate(file_paths):
            try:
                # Update progress
                progress = int((idx / total_files) * 100)
                self.progress_update.emit(progress, f"Processing {file_path.name}...")
                
                # 1. Detect Format
                file_fmt, conf = self.format_detector.detect(file_path)
                
                # 2. Select Decoder
                decoder = next((d for d in self.decoders if d.can_decode(file_path)), None)
                
                if not decoder:
                    # Fallback for txt if unknown but looks text-like? 
                    # For now just log error
                    errors.append({
                        "file": str(file_path),
                        "error": f"No suitable decoder found for format {file_fmt}"
                    })
                    continue
                    
                # 3. Decode
                result = decoder.decode(file_path)
                
                if result.get('errors'):
                    for err in result['errors']:
                        errors.append({"file": str(file_path), "error": err})
                
                # 4. Process Sections -> Segments
                # We treat each section (or the whole text) as a potential segment
                # If sections are available, use them. Otherwise use full text.
                items_to_process = result.get('sections', [])
                if not items_to_process and result.get('text'):
                    items_to_process = [{'content': result['text'], 'type': 'full_text'}]
                
                for item in items_to_process:
                    content = item.get('content', '').strip()
                    if not content:
                        continue
                    
                    # Normalize
                    normalized_content = self.normalizer.normalize(content)
                    
                    # Detect Language
                    lang, lang_conf = self.language_detector.detect(normalized_content)
                    
                    # Update stats
                    language_counts[lang] = language_counts.get(lang, 0) + 1
                    
                    # Create Segment
                    segment = TextSegment(
                        id=str(uuid.uuid4()),
                        content=normalized_content,
                        language=lang,
                        language_confidence=lang_conf,
                        source_file=file_path.name,
                        source_format=file_fmt,
                        page_or_section=str(item.get('number', item.get('heading', ''))),
                        word_count=len(normalized_content.split()),
                        char_count=len(normalized_content)
                    )
                    
                    segments.append(segment)
                    
            except Exception as e:
                self.logger.error(f"Failed to process {file_path}: {e}")
                errors.append({"file": str(file_path), "error": str(e)})
        
        # Finalize
        self.progress_update.emit(100, "Ingestion complete")
        
        result_obj = IngestionResult(
            project_id=str(uuid.uuid4()), # New project ID
            timestamp=datetime.now(),
            total_files_processed=total_files,
            total_segments=len(segments),
            segments=segments,
            language_distribution=language_counts,
            errors=errors
        )
        
        self.stage_completed.emit(result_obj)
        return result_obj
