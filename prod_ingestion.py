# backend/app/services/ingestion/chunker.py
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
import hashlib
import re

class SmartChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 128):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Hierarchical separators
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self._count_tokens,
            separators=[
                "\n\n\n",  # Multiple newlines
                "\n\n",    # Paragraph breaks
                "\n",      # Single newlines
                ". ",      # Sentences
                "! ",
                "? ",
                "; ",
                ", ",
                " ",       # Words
                ""
            ]
        )
    
    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))
    
    def _extract_metadata(self, text: str, chunk_index: int) -> Dict:
        """Extract metadata from chunk"""
        return {
            "num_tokens": self._count_tokens(text),
            "num_chars": len(text),
            "num_words": len(text.split()),
            "has_code": bool(re.search(r'```|def |class |import ', text)),
            "has_url": bool(re.search(r'https?://', text)),
            "chunk_index": chunk_index
        }
    
    def chunk_text(
        self, 
        text: str, 
        source: str,
        preserve_structure: bool = True
    ) -> List[Dict]:
        """
        Chunk text with context preservation
        
        Args:
            text: Input text
            source: Document source identifier
            preserve_structure: Keep section headers in chunks
        """
        if not text or len(text.strip()) < 50:
            return []
        
        # Clean text
        text = self._clean_text(text)
        
        # Split into chunks
        chunks = self.splitter.split_text(text)
        
        result = []
        for idx, chunk in enumerate(chunks):
            # Add context from previous chunk
            context_prefix = ""
            if idx > 0 and preserve_structure:
                prev_chunk = chunks[idx - 1]
                # Extract last sentence of previous chunk
                sentences = prev_chunk.split(". ")
                if len(sentences) > 0:
                    context_prefix = f"[Context: ...{sentences[-1]}]\n\n"
            
            chunk_text = context_prefix + chunk
            
            chunk_dict = {
                "id": hashlib.md5(chunk_text.encode()).hexdigest(),
                "text": chunk_text,
                "metadata": {
                    "source": source,
                    **self._extract_metadata(chunk_text, idx),
                    "has_context": bool(context_prefix)
                }
            }
            result.append(chunk_dict)
        
        return result
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove control characters
        text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        return text.strip()

# backend/app/services/ingestion/preprocessor.py
from typing import Dict, List
import re
from bs4 import BeautifulSoup

class TextPreprocessor:
    """Advanced text preprocessing"""
    
    @staticmethod
    def clean_html(html_content: str) -> str:
        """Remove HTML tags and extract clean text"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Break into lines and remove leading/trailing space
        lines = (line.strip() for line in text.splitlines())
        
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        
        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    
    @staticmethod
    def extract_metadata_from_text(text: str) -> Dict:
        """Extract structured metadata from text"""
        metadata = {}
        
        # Extract emails
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        if emails:
            metadata['emails'] = list(set(emails))
        
        # Extract phone numbers
        phones = re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text)
        if phones:
            metadata['phone_numbers'] = list(set(phones))
        
        # Extract dates
        dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text)
        if dates:
            metadata['dates'] = list(set(dates))
        
        # Detect language (simple heuristic)
        if re.search(r'[а-яА-Я]', text):
            metadata['detected_language'] = 'ru'
        elif re.search(r'[à-ÿÀ-Ÿ]', text):
            metadata['detected_language'] = 'fr'
        else:
            metadata['detected_language'] = 'en'
        
        return metadata
    
    @staticmethod
    def remove_pii(text: str, redact: bool = True) -> str:
        """Remove or redact PII (Personal Identifiable Information)"""
        if not redact:
            return text
        
        # Redact emails
        text = re.sub(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            '[EMAIL_REDACTED]',
            text
        )
        
        # Redact phone numbers
        text = re.sub(
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            '[PHONE_REDACTED]',
            text
        )
        
        # Redact SSN-like patterns
        text = re.sub(
            r'\b\d{3}-\d{2}-\d{4}\b',
            '[SSN_REDACTED]',
            text
        )
        
        return text

# backend/app/services/ingestion/parsers/pdf.py
import PyPDF2
from pdf2image import convert_from_path
import pytesseract
from typing import Dict, Optional
import io

class PDFParser:
    """Advanced PDF parsing with OCR fallback"""
    
    @staticmethod
    def parse(file_path: str, enable_ocr: bool = True) -> Dict:
        """
        Parse PDF with text extraction and OCR fallback
        
        Returns:
            {
                'text': str,
                'metadata': dict,
                'num_pages': int,
                'used_ocr': bool
            }
        """
        result = {
            'text': '',
            'metadata': {},
            'num_pages': 0,
            'used_ocr': False
        }
        
        try:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                result['num_pages'] = len(reader.pages)
                
                # Extract metadata
                if reader.metadata:
                    result['metadata'] = {
                        'title': reader.metadata.get('/Title', ''),
                        'author': reader.metadata.get('/Author', ''),
                        'subject': reader.metadata.get('/Subject', ''),
                        'creator': reader.metadata.get('/Creator', ''),
                        'producer': reader.metadata.get('/Producer', ''),
                        'creation_date': str(reader.metadata.get('/CreationDate', ''))
                    }
                
                # Extract text
                text_parts = []
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    
                    # If text extraction yields little content, try OCR
                    if enable_ocr and len(page_text.strip()) < 50:
                        try:
                            ocr_text = PDFParser._ocr_page(file_path, page_num)
                            text_parts.append(ocr_text)
                            result['used_ocr'] = True
                        except Exception:
                            text_parts.append(page_text)
                    else:
                        text_parts.append(page_text)
                
                result['text'] = '\n\n'.join(text_parts)
        
        except Exception as e:
            raise ValueError(f"Failed to parse PDF: {str(e)}")
        
        return result
    
    @staticmethod
    def _ocr_page(pdf_path: str, page_num: int) -> str:
        """Perform OCR on a specific page"""
        images = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1)
        if images:
            return pytesseract.image_to_string(images[0])
        return ""