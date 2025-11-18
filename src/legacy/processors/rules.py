import pandas as pd
from pathlib import Path


class CSVProcessor:
    def __init__(self, input_dir: str = "data/csv"):
        self.input_dir = Path(input_dir)
        # Column consolidation rules: (col1_idx, col2_idx) -> new_name
        self.rules = {
            (0, 1): "분류-대-공정",
            (2, 3): "분류-중-재료",
            (4, 5): "분류-소-객체유형"
        }

    def consolidate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        processed_df = df.copy()
        columns_to_remove = []

        for (col1_idx, col2_idx), new_name in self.rules.items():
            if col1_idx < len(df.columns) and col2_idx < len(df.columns):
                # Consolidate values with '-' separator
                consolidated = []
                for i in range(len(df)):
                    val1 = str(df.iloc[i, col1_idx]).strip() if pd.notna(
                        df.iloc[i, col1_idx]) else ""
                    val2 = str(df.iloc[i, col2_idx]).strip() if pd.notna(
                        df.iloc[i, col2_idx]) else ""

                    val1 = "" if val1.lower() == "nan" else val1
                    val2 = "" if val2.lower() == "nan" else val2

                    if val1 and val2:
                        consolidated.append(f"{val1}-{val2}")
                    elif val1 or val2:
                        consolidated.append(val1 or val2)
                    else:
                        consolidated.append("")

                # Update column with proper dtype handling
                processed_df.iloc[:, col1_idx] = consolidated
                columns_to_remove.append(col2_idx)

        # Remove duplicate columns and rename
        for col_idx in sorted(columns_to_remove, reverse=True):
            processed_df = processed_df.drop(
                processed_df.columns[col_idx], axis=1)

        # Rename the consolidated columns
        for i, ((col1_idx, col2_idx), new_name) in enumerate(self.rules.items()):
            if col1_idx < len(processed_df.columns):
                # Adjust index after column removals
                adjusted_idx = col1_idx - \
                    sum(1 for idx in columns_to_remove if idx < col1_idx)
                if 0 <= adjusted_idx < len(processed_df.columns):
                    processed_df.columns.values[adjusted_idx] = new_name

        return processed_df

    def remove_empty_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows that are empty or have only one non-empty value.

        Examples:
        - ,,,,,,,,,,, -> remove (0 values)
        - ,,,,,,,,,E06, -> remove (1 value)  
        - ,,,,,108,15,,,, -> keep (2 values)
        """
        if df.empty:
            return df

        # Count non-empty values per row (excluding NaN and empty strings)
        non_empty_counts = df.apply(
            lambda row: sum(
                1 for val in row
                if pd.notna(val) and str(val).strip() != '' and str(val).lower() != 'nan'
            ),
            axis=1
        )

        # Keep only rows with 2 or more non-empty values
        return df[non_empty_counts >= 2].reset_index(drop=True)

    def process_file(self, file_path: Path) -> bool:
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            if df.empty:
                return False

            processed_df = self.consolidate_columns(df)
            processed_df = self.remove_empty_rows(processed_df)
            processed_df.to_csv(file_path, index=False, encoding='utf-8')
            return True
        except Exception as e:
            print(f"❌ Error processing {file_path.name}: {e}")
            return False

    def process_all(self) -> dict:
        csv_files = list(self.input_dir.glob("KBIMS_*.csv"))
        if not csv_files:
            print(f"❌ No KBIMS CSV files found in {self.input_dir}")
            return {}

        results = {}
        for file_path in csv_files:
            results[file_path.name] = self.process_file(file_path)

        return results


def main():
    processor = CSVProcessor()

    if not processor.input_dir.exists():
        print(f"❌ Directory not found: {processor.input_dir}")
        return

    results = processor.process_all()
    if not results:
        return

    successful = sum(results.values())
    failed = len(results) - successful

    print(f"✅ Processed: {successful}/{len(results)} files")
    if failed > 0:
        print(f"❌ Failed: {failed} files")


if __name__ == "__main__":
    main()
