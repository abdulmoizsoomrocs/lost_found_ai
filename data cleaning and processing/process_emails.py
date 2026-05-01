"""
Lost & Found Email Processing Pipeline
Cleans unstructured email data and extracts structured information.
"""

import pandas as pd
import numpy as np
import re
import logging
from pathlib import Path
from typing import Optional, Dict, List, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmailProcessor:
    """Production-quality email processing pipeline for lost & found data."""

    # Category keywords mapping
    CATEGORY_KEYWORDS = {
        'wallet': ['wallet'],
        'phone': ['phone', 'iphone', 'mobile'],
        'keys': ['key', 'keys'],
        'bag': ['bag', 'backpack'],
        'earbuds': ['earbuds', 'airbuds'],
        'glasses': ['glasses'],
        'watch': ['watch'],
        'usb': ['usb'],
        'id_card': ['id', 'card']
    }

    # Location patterns
    LOCATION_PATTERNS = {
        'library': r'\blibrary\b',
        'block': r'\bblock\s+(\d+)\b',
        'ab': r'\bab\s*(\d+)\b',
        'parking': r'\bparking\b',
        'hall': r'\bhall\b',
        'washroom': r'\bwashroom\b',
        'ground': r'\bground\b'
    }

    def __init__(self, input_file: str = 'emails_output.csv'):
        """Initialize processor with input file path."""
        self.input_file = input_file
        self.df = None
        self.initial_row_count = 0
        self.final_row_count = 0

    def load_data(self) -> bool:
        """Load CSV file with error handling."""
        try:
            logger.info(f"Loading data from {self.input_file}")
            self.df = pd.read_csv(self.input_file)
            
            # Validate required columns
            required_columns = {'subject', 'body'}
            if not required_columns.issubset(self.df.columns):
                raise ValueError(f"Missing required columns: {required_columns - set(self.df.columns)}")
            
            self.initial_row_count = len(self.df)
            logger.info(f"Loaded {self.initial_row_count} rows")
            
            # Fill missing values
            self.df['subject'] = self.df['subject'].fillna('')
            self.df['body'] = self.df['body'].fillna('')
            
            # Create date column if it doesn't exist, will be extracted from email metadata
            if 'date' not in self.df.columns:
                self.df['date'] = ''
            else:
                self.df['date'] = self.df['date'].fillna('')
            
            return True
        except FileNotFoundError:
            logger.error(f"File not found: {self.input_file}")
            return False
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False

    @staticmethod
    def _remove_forwarded_headers(text: str) -> str:
        """Remove forwarded email headers."""
        # Remove forwarded message separator
        text = re.sub(r'-+\s*Forwarded message\s*-+.*?(?=\n|$)', '', text, flags=re.IGNORECASE | re.DOTALL)
        return text

    @staticmethod
    def _remove_email_metadata(text: str) -> str:
        """Remove From:, To:, Date: lines."""
        lines = text.split('\n')
        filtered_lines = [
            line for line in lines
            if not re.match(r'^\s*(From|To|Date|Cc|Bcc):\s*', line, re.IGNORECASE)
        ]
        return '\n'.join(filtered_lines)

    @staticmethod
    def _remove_email_addresses(text: str) -> str:
        """Remove email addresses from text."""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.sub(email_pattern, '', text)

    @staticmethod
    def _remove_phone_numbers(text: str) -> str:
        """Remove phone numbers in various formats."""
        # Match patterns like: (123) 456-7890, 123-456-7890, 123.456.7890, +1-123-456-7890, etc.
        phone_patterns = [
            r'\+?\d{1,3}\s*\(?\d{3}\)?\s*[\s.-]?\d{3}[\s.-]?\d{4}\b',  # International or standard formats
            r'\b\d{10}\b',  # 10 digit number without separators
            r'\(\d{3}\)\s*\d{3}[-.\s]?\d{4}\b',  # (123) 456-7890 format
        ]
        for pattern in phone_patterns:
            text = re.sub(pattern, '', text)
        return text

    @staticmethod
    def _remove_special_characters(text: str) -> str:
        """Remove special characters but keep spaces and basic punctuation needed for readability."""
        # Keep alphanumeric and spaces, replace special characters with space to preserve word boundaries
        text = re.sub(r'[^a-z0-9\s]', ' ', text, flags=re.IGNORECASE)
        return text

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Convert to lowercase and remove extra spaces."""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
        return text.strip()

    @staticmethod
    def _compress_text(text: str, max_words: int = 10) -> str:
        """Compress text to keep only item names, colors, and descriptive words.
        
        Removes: location words, lost/found, stopwords, fillers
        Keeps: item names, colors, brands, models, numbers (64gb, etc.)
        
        Args:
            text: The text to compress
            max_words: Maximum number of words to keep (default: 10)
        
        Returns:
            Compressed text with only meaningful descriptive content
        """
        if not isinstance(text, str) or len(text) == 0:
            return ""
        
        # Comprehensive stopwords and words to remove
        stopwords = {
            # Common words
            'i', 'my', 'the', 'is', 'am', 'are', 'was', 'were', 'have', 'has', 'had',
            'hello', 'dear', 'please', 'kindly', 'thank', 'thanks', 'subject', 'fwd',
            'front', 'desk', 'officer', 'university', 'contact', 'number', 'a', 'an',
            'and', 'or', 'but', 'if', 'at', 'to', 'for', 'of', 'in', 'on', 'with',
            'by', 'this', 'that', 'be', 'you', 'your', 'we', 'our', 'it', 'so',
            'as', 'from', 'do', 'did', 'will', 'would', 'can', 'could', 'should',
            'may', 'might', 'extension', 'direct', 'reception', 'receptionist',
            # Status words to exclude
            'lost', 'found', 'lose', 'find',
            # Location words to exclude
            'library', 'block', 'ab', 'parking', 'washroom', 'hall', 'ground',
            'room', 'outside', 'near', 'area', 'gate', 'campus', 'cafe', 'masjid',
            'hostel', 'auditorium', 'point', 'wazu', 'khana', 'table', 'desk',
            'shelf', 'pocket', 'floor', 'lab', 'cc', 'abv', 'old', 'new',
            # Other fillers
            'hope', 'email', 'whatsapp', 'cell', 'tell', 'anyone', 'someone',
            'kindly', 'please', 'await', 'regards', 'thanks', 'thank', 'collect',
            'happened', 'happened', 'yesterday', 'today', 'tomorrow', 'week', 'day',
            'month', 'year', 'time', 'morning', 'evening', 'night', 'around',
            'also', 'only', 'still', 'just', 'back', 'best'
        }
        
        words = text.split()
        
        # Remove stopwords and duplicates while preserving order
        seen = set()
        filtered_words = []
        for word in words:
            # Skip stopwords and duplicates
            if word not in stopwords and word not in seen:
                filtered_words.append(word)
                seen.add(word)
        
        # Keep only first max_words (5-10) important words
        return ' '.join(filtered_words[:max_words])

    def _refine_clean_text(self, text: str, category: str = 'unknown') -> str:
        """Refine clean text to product-level descriptors only (3-6 words).
        
        Keeps only: item names, colors, brand/model, size/storage
        Removes: filler words, verbs, adjectives, time references
        
        Args:
            text: The compressed text
            category: The extracted category (for item name reference)
        
        Returns:
            Ultra-compact product-like description (3-6 words)
        """
        if not isinstance(text, str) or len(text) == 0:
            return ""
        
        # Keywords by category
        colors = {
            'black', 'white', 'grey', 'gray', 'red', 'blue', 'green', 'yellow',
            'silver', 'gold', 'brown', 'pink', 'purple', 'orange', 'beige',
            'navy', 'maroon', 'teal', 'turquoise', 'golden', 'dark', 'light'
        }
        
        brands_models = {
            'iphone', 'samsung', 'apple', 'dell', 'hp', 'lenovo', 'sony', 'canon',
            'ronin', 'airbuds', 'airpods', 'realme', 'oppo', 'vivo', 'infinix',
            'audionic', 'headphones', 'wireless', 'pro', 'max', 'plus', 'lite',
            'mini', 'ultra', 'standard', 'model', 'series', 'edition', 'compact',
            'thinkpad', 'probook', 'pavilion'
        }
        
        sizes_storage = {
            '64gb', '32gb', '128gb', '256gb', '512gb', '1tb',
            '15', '14', '13', '12', '11', '10', '9', '8', '7', '6', '5',
            'gb', 'mb', 'tb', 'inch'
        }
        
        # Item-specific keywords to preserve
        item_keywords = {
            'wallet': ['wallet', 'purse'],
            'phone': ['phone', 'mobile', 'iphone', 'oppo', 'samsung', 'infinix', 'vivo', 'realme'],
            'keys': ['key', 'keys', 'keychain'],
            'bag': ['bag', 'backpack', 'shopper', 'purse', 'beg'],
            'earbuds': ['earbud', 'earbuds', 'earphone', 'airbuds', 'airpods', 'headphone', 'headphones'],
            'glasses': ['glass', 'glasses', 'eyeglasses', 'spectacles'],
            'watch': ['watch', 'smartwatch'],
            'usb': ['usb', 'pendrive', 'stick', 'drive'],
            'id_card': ['card', 'id', 'cnic', 'atm']
        }
        
        words = text.split()
        important_words = []
        
        # First pass: collect important words by priority
        item_name_found = False
        for word in words:
            # Skip very short words (1 char) and common fillers
            if len(word) < 2 or word in {'a', 'i', 'pm', 'am'}:
                continue
            
            # Priority 1: Item name from category
            if not item_name_found and category in item_keywords:
                if word in item_keywords[category]:
                    important_words.append(word)
                    item_name_found = True
                    continue
            
            # Priority 2: Colors
            if word in colors:
                important_words.append(word)
                continue
            
            # Priority 3: Brand/model/size
            if word in brands_models or word in sizes_storage:
                important_words.append(word)
                continue
        
        # If no item name found from category, try generic item keywords
        if not item_name_found:
            all_item_keywords = set()
            for keywords_list in item_keywords.values():
                all_item_keywords.update(keywords_list)
            
            for word in words:
                if word in all_item_keywords:
                    important_words.insert(0, word)
                    item_name_found = True
                    break
        
        # Remove duplicates while preserving order
        seen = set()
        final_words = []
        for word in important_words:
            if word not in seen:
                final_words.append(word)
                seen.add(word)
        
        # Limit to max 6 words
        result = ' '.join(final_words[:6])
        return result if result else ""

    def clean_text(self, text: str) -> str:
        """Apply all cleaning operations in sequence."""
        if not isinstance(text, str):
            return ""
        
        text = self._remove_forwarded_headers(text)
        text = self._remove_email_metadata(text)
        text = self._remove_email_addresses(text)
        text = self._remove_phone_numbers(text)
        text = self._remove_special_characters(text)
        text = self._normalize_whitespace(text)
        text = self._compress_text(text)  # Add compression to keep only important words
        
        return text

    def extract_category(self, text: str) -> str:
        """Extract category based on keyword matching with word boundaries."""
        if not isinstance(text, str) or len(text) == 0:
            return 'unknown'
        
        text_lower = f" {text.lower()} "
        
        # Check each category's keywords with word boundaries
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if f" {keyword} " in text_lower:
                    return category
        
        return 'unknown'

    def extract_location(self, text: str) -> Optional[str]:
        """Extract location using priority-based matching.
        
        Priority 1: Specific locations with numbers (block X, abX)
        Priority 2: General locations (library, parking, hall, washroom, ground)
        """
        if not isinstance(text, str) or len(text) == 0:
            return None
        
        text_lower = text.lower()
        
        # Priority 1: Detect specific locations with numbers
        # Match "block X" pattern
        block_match = re.search(r'\bblock\s+(\d+)\b', text_lower)
        if block_match:
            return f"block {block_match.group(1)}"
        
        # Match "ab X" or "abX" pattern
        ab_match = re.search(r'\bab\s*(\d+)\b', text_lower)
        if ab_match:
            return f"ab{ab_match.group(1)}"
        
        # Priority 2: Detect general locations
        general_locations = {
            'library': r'\blibrary\b',
            'parking': r'\bparking\b',
            'hall': r'\bhall\b',
            'washroom': r'\bwashroom\b',
            'ground': r'\bground\b'
        }
        
        for location_name, pattern in general_locations.items():
            if re.search(pattern, text_lower):
                return location_name
        
        return None

    @staticmethod
    def _extract_date_from_metadata(text: str) -> Optional[str]:
        """Extract date from email metadata in the text."""
        if not isinstance(text, str):
            return None
        
        # Look for "Date: ..." pattern in email headers
        date_match = re.search(r'Date:\s*(.+?)(?=\n|Subject:|$)', text, re.IGNORECASE)
        if date_match:
            return date_match.group(1).strip()
        return None

    def extract_status(self, text: str) -> str:
        """Extract status (lost or found) from text.
        
        Returns:
            "lost" if text contains "lost"
            "found" if text contains "found"
            "unknown" otherwise
        """
        if not isinstance(text, str):
            return 'unknown'
        
        text_lower = text.lower()
        
        # Check for "lost" first
        if ' lost ' in f" {text_lower} ":
            return 'lost'
        # Check for "found"
        if ' found ' in f" {text_lower} ":
            return 'found'
        
        return 'unknown'

    def process(self) -> bool:
        """Execute the complete processing pipeline."""
        try:
            logger.info("Starting data processing pipeline...")
            
            # Step 1: Combine subject and body
            logger.info("Step 1: Combining subject and body...")
            self.df['text'] = self.df['subject'].fillna('') + ' ' + self.df['body'].fillna('')
            
            # Step 2: Clean text
            logger.info("Step 2: Cleaning text...")
            self.df['clean_text'] = self.df['text'].apply(self.clean_text)
            
            # Step 3: Extract category
            logger.info("Step 3: Extracting category...")
            self.df['category'] = self.df['clean_text'].apply(self.extract_category)
            
            # Step 3b: Refine clean_text using category knowledge
            logger.info("Step 3b: Refining clean_text to product-level descriptors...")
            self.df['clean_text'] = self.df.apply(lambda row: self._refine_clean_text(row['clean_text'], row['category']), axis=1)
            
            # Step 4: Extract location (use original text, not cleaned)
            logger.info("Step 4: Extracting location...")
            self.df['location'] = self.df['text'].apply(self.extract_location)
            
            # Step 5: Extract status (lost or found)
            logger.info("Step 5: Extracting status...")
            self.df['status'] = self.df['text'].apply(self.extract_status)
            
            # Step 6: Extract date from email metadata if not already present
            logger.info("Step 6: Processing dates...")
            # If date column is empty, try to extract from email body
            empty_dates = self.df['date'] == ''
            if empty_dates.any():
                self.df.loc[empty_dates, 'date'] = self.df.loc[empty_dates, 'text'].apply(self._extract_date_from_metadata)
            
            # Convert date to datetime
            self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
            
            # Step 7: Keep all categories including unknown for analysis
            # logger.info("Step 7: Filtering unknown categories...")
            # initial_filtered = len(self.df)
            # self.df = self.df[self.df['category'] != 'unknown'].copy()
            # filtered_count = len(self.df)
            # logger.info(f"Removed {initial_filtered - filtered_count} rows with unknown category")
            
            # Step 8: Select final columns
            logger.info("Step 8: Selecting final columns...")
            self.df = self.df[['clean_text', 'category', 'location', 'status', 'date']]
            
            self.final_row_count = len(self.df)
            logger.info(f"Processing complete. {self.final_row_count} rows remain.")
            
            return True
        except Exception as e:
            logger.error(f"Error during processing: {e}")
            return False

    def save_output(self, output_file: str = 'final_dataset.csv') -> bool:
        """Save processed data to CSV."""
        try:
            logger.info(f"Saving output to {output_file}")
            self.df.to_csv(output_file, index=False)
            logger.info(f"Successfully saved {len(self.df)} rows to {output_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving output: {e}")
            return False

    def print_summary(self):
        """Print processing summary and sample data."""
        print("\n" + "="*70)
        print("LOST & FOUND EMAIL PROCESSING SUMMARY")
        print("="*70)
        print(f"\nTotal rows before cleaning:  {self.initial_row_count}")
        print(f"Total rows after filtering:  {self.final_row_count}")
        print(f"Rows removed:                {self.initial_row_count - self.final_row_count}")
        
        if self.final_row_count > 0:
            print("\n" + "-"*70)
            print("CATEGORY DISTRIBUTION:")
            print("-"*70)
            print(self.df['category'].value_counts().to_string())
            
            print("\n" + "-"*70)
            print("LOCATION DISTRIBUTION:")
            print("-"*70)
            location_counts = self.df['location'].value_counts(dropna=True)
            if len(location_counts) > 0:
                print(location_counts.to_string())
            else:
                print("No locations extracted")
            
            print("\n" + "-"*70)
            print("SAMPLE OF 5 CLEANED ROWS:")
            print("-"*70)
            sample_df = self.df.head(5).copy()
            for idx, row in sample_df.iterrows():
                print(f"\nRow {idx + 1}:")
                print(f"  Category: {row['category']}")
                print(f"  Location: {row['location']}")
                print(f"  Date:     {row['date']}")
                print(f"  Text:     {row['clean_text'][:100]}..." if len(row['clean_text']) > 100 else f"  Text:     {row['clean_text']}")
        
        print("\n" + "="*70 + "\n")

    def run(self) -> bool:
        """Execute the complete pipeline."""
        if not self.load_data():
            return False
        
        if not self.process():
            return False
        
        if not self.save_output():
            return False
        
        self.print_summary()
        return True


def main():
    """Main entry point."""
    try:
        processor = EmailProcessor('emails_output.csv')
        success = processor.run()
        
        if success:
            logger.info("Pipeline completed successfully!")
            return 0
        else:
            logger.error("Pipeline failed!")
            return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
