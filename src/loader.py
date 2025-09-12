import pandas as pd
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders.csv_loader import CSVLoader
from tqdm import tqdm

from config import CSV_DIR, CSV_FILE_PATTERN, CSV_ENCODING


class KBIMSCSVLoader:
    """
    Specialized loader for KBIMS (Korea Building Information Modeling Standard) CSV files.
    Handles Korean text and building component hierarchical classification system.
    """

    def __init__(self, csv_directory: str = None, encoding: str = CSV_ENCODING):
        """
        Initialize KBIMS CSV Loader.

        Args:
            csv_directory: Directory containing KBIMS CSV files
            encoding: Text encoding for CSV files (default: utf-8 for Korean)
        """
        self.csv_directory = Path(csv_directory or CSV_DIR)
        self.encoding = encoding
        self.csv_files = list(self.csv_directory.glob(CSV_FILE_PATTERN))

        if not self.csv_files:
            raise FileNotFoundError(
                f"No CSV files found matching pattern '{CSV_FILE_PATTERN}' in {self.csv_directory}")

    def load_all_documents(self) -> List[Document]:
        """
        Load all KBIMS CSV files and return as LangChain Documents.

        Returns:
            List of Document objects with metadata
        """
        all_documents = []

        print(f"Loading {len(self.csv_files)} KBIMS CSV files...")

        for csv_file in tqdm(self.csv_files, desc="Processing CSV files"):
            try:
                documents = self._load_csv_file(csv_file)
                all_documents.extend(documents)
            except Exception as e:
                print(f"Error loading {csv_file.name}: {e}")
                continue

        print(
            f"Successfully loaded {len(all_documents)} documents from {len(self.csv_files)} CSV files")
        return all_documents

    def _load_csv_file(self, csv_file_path: Path) -> List[Document]:
        """
        Load a single CSV file and convert rows to Documents.

        Args:
            csv_file_path: Path to CSV file

        Returns:
            List of Document objects
        """
        # Extract category from filename
        category = self._extract_category_from_filename(csv_file_path.name)

        # Read CSV with pandas for better Korean text handling
        df = pd.read_csv(csv_file_path, encoding=self.encoding)

        # Clean column names (remove whitespace)
        df.columns = df.columns.str.strip()

        documents = []
        for idx, row in df.iterrows():
            # Skip empty rows
            if row.isna().all() or self._is_empty_row(row):
                continue

            doc = self._create_document_from_row(
                row, category, csv_file_path.name, idx)
            if doc:
                documents.append(doc)

        return documents

    def _create_document_from_row(self, row: pd.Series, category: str, filename: str, row_index: int) -> Optional[Document]:
        """
        Convert CSV row to LangChain Document.

        Args:
            row: Pandas Series representing CSV row
            category: Building component category
            filename: Source CSV filename
            row_index: Row index in CSV

        Returns:
            Document object or None if row is invalid
        """
        # Create meaningful content by combining key fields
        content_parts = []

        # Add classification hierarchy
        classification_fields = [
            '분류-대-공정', '분류-중-재료', '분류-소-객체유형'
        ]

        for field in classification_fields:
            if field in row and pd.notna(row[field]) and str(row[field]).strip():
                content_parts.append(f"{field}: {row[field]}")

        # Add library information
        library_fields = [
            '라이브러리-라이브러리 ID(코드)', '라이브러리-파일명',
            '라이브러리-명칭', '라이브러리-명칭(유형명)'
        ]

        for field in library_fields:
            if field in row and pd.notna(row[field]) and str(row[field]).strip():
                content_parts.append(f"{field}: {row[field]}")

        # Add standard codes
        if '조달청 표준공사코드-세부공종코드(전체)' in row and pd.notna(row['조달청 표준공사코드-세부공종코드(전체)']):
            content_parts.append(f"조달청 표준공사코드: {row['조달청 표준공사코드-세부공종코드(전체)']}")

        # Add KBIMS classification
        if '관련규격-KBIMS 부위분류' in row and pd.notna(row['관련규격-KBIMS 부위분류']):
            content_parts.append(f"KBIMS 부위분류: {row['관련규격-KBIMS 부위분류']}")

        # Add notes if available
        if '비고' in row and pd.notna(row['비고']) and str(row['비고']).strip():
            content_parts.append(f"비고: {row['비고']}")

        if not content_parts:
            return None

        # Combine content parts
        content = "\n".join(content_parts)

        # Create metadata
        metadata = {
            'source': filename,
            'category': category,
            'row_index': row_index,
            'classification_major': self._safe_get_field(row, '분류-대-공정'),
            'classification_medium': self._safe_get_field(row, '분류-중-재료'),
            'classification_minor': self._safe_get_field(row, '분류-소-객체유형'),
            'library_id': self._safe_get_field(row, '라이브러리-라이브러리 ID(코드)'),
            'library_name': self._safe_get_field(row, '라이브러리-명칭'),
            'standard_code': self._safe_get_field(row, '조달청 표준공사코드-세부공종코드(전체)'),
            'kbims_classification': self._safe_get_field(row, '관련규격-KBIMS 부위분류')
        }

        # Remove None values from metadata
        metadata = {k: v for k, v in metadata.items() if v is not None}

        return Document(page_content=content, metadata=metadata)

    def _extract_category_from_filename(self, filename: str) -> str:
        """
        Extract building component category from CSV filename.

        Args:
            filename: CSV filename

        Returns:
            Category name in Korean
        """
        # Remove KBIMS_ prefix and .csv suffix, then extract category
        if filename.startswith('KBIMS_'):
            category_part = filename[6:-4]  # Remove 'KBIMS_' and '.csv'
            # Split by '-' and take the part after the number
            if '-' in category_part:
                return category_part.split('-', 1)[1]
            return category_part
        return filename[:-4]  # Just remove .csv

    def _safe_get_field(self, row: pd.Series, field_name: str) -> Optional[str]:
        """
        Safely get field value from row, handling NaN and empty strings.

        Args:
            row: Pandas Series
            field_name: Field name to extract

        Returns:
            Field value or None if empty/invalid
        """
        if field_name not in row:
            return None

        value = row[field_name]
        if pd.isna(value):
            return None

        str_value = str(value).strip()
        return str_value if str_value else None

    def _is_empty_row(self, row: pd.Series) -> bool:
        """
        Check if row is effectively empty (has fewer than 2 meaningful values).

        Args:
            row: Pandas Series representing CSV row

        Returns:
            True if row is empty or has insufficient data
        """
        non_empty_count = 0
        for value in row:
            if pd.notna(value) and str(value).strip():
                non_empty_count += 1
                if non_empty_count >= 2:
                    return False
        return True

    def get_documents_by_category(self, category: str) -> List[Document]:
        """
        Load documents for a specific building component category.

        Args:
            category: Category name (e.g., "기초", "벽", "지붕")

        Returns:
            List of Document objects for the specified category
        """
        matching_files = [f for f in self.csv_files if category in f.name]

        if not matching_files:
            print(f"No files found for category: {category}")
            return []

        documents = []
        for csv_file in matching_files:
            try:
                file_documents = self._load_csv_file(csv_file)
                documents.extend(file_documents)
            except Exception as e:
                print(f"Error loading {csv_file.name}: {e}")
                continue

        return documents

    def get_available_categories(self) -> List[str]:
        """
        Get list of available building component categories.

        Returns:
            List of category names
        """
        categories = []
        for csv_file in self.csv_files:
            category = self._extract_category_from_filename(csv_file.name)
            if category not in categories:
                categories.append(category)
        return sorted(categories)

    def load_documents_with_filter(self, **filters) -> List[Document]:
        """
        Load documents with metadata filtering.

        Args:
            **filters: Metadata filters (e.g., classification_major="FT000-기초")

        Returns:
            Filtered list of Document objects
        """
        all_documents = self.load_all_documents()

        if not filters:
            return all_documents

        filtered_documents = []
        for doc in all_documents:
            match = True
            for key, value in filters.items():
                if key not in doc.metadata or doc.metadata[key] != value:
                    match = False
                    break
            if match:
                filtered_documents.append(doc)

        print(
            f"Filtered {len(all_documents)} documents down to {len(filtered_documents)} documents")
        return filtered_documents


def load_kbims_documents(csv_directory: str = None) -> List[Document]:
    """
    Convenience function to load all KBIMS documents.

    Args:
        csv_directory: Directory containing KBIMS CSV files

    Returns:
        List of Document objects
    """
    loader = KBIMSCSVLoader(csv_directory)
    return loader.load_all_documents()


if __name__ == "__main__":
    # Test the loader
    loader = KBIMSCSVLoader()

    print("Available categories:")
    for category in loader.get_available_categories():
        print(f"  - {category}")

    print("\nLoading all documents...")
    documents = loader.load_all_documents()

    print(f"\nLoaded {len(documents)} documents")

    if documents:
        print("\nSample document:")
        print("Content:", documents[0].page_content[:200] + "...")
        print("Metadata:", documents[0].metadata)
