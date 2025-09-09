import pandas as pd
import os

SHEET_SKIP_ROWS = {
    '03-기초': 9,  # Data starts at row 10 (0-indexed)
    '04-기둥': 4,  # Data starts at row 5 (0-indexed)
    '05-보': 4,    # Data starts at row 5 (0-indexed)
    '06-바닥': 4,  # Data starts at row 5 (0-indexed)
    '08-천장': 4,  # Likely same pattern
    '07-벽': 4,    # Likely same pattern
    '09-지붕': 4,  # Likely same pattern
    '10-문': 4,    # Likely same pattern
    '11-창': 4,    # Likely same pattern
    '12-커튼월': 4,  # Likely same pattern
    '13-계단': 4,  # Likely same pattern
    '14-램프': 4,  # Likely same pattern
    '15-난간': 4,  # Likely same pattern
    '16-가구 및 장비': 4,  # Likely same pattern
    '17-위생설비': 4,     # Likely same pattern
    '18-조경': 4,         # Likely same pattern
    '19-운송설비': 4,     # Likely same pattern
    '20-주석기호': 4,     # Likely same pattern
    '21-기타 환경(토목)': 4,  # Likely same pattern
}

# Selected columns to extract from Excel (B,C,D,E,F,G,H,J,K,L,N,O,R,S,Z,AL)
# Using numeric indices (0-based) instead of letters to avoid column existence errors
SELECTED_COLUMNS = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 13,
                    14, 17, 18, 25, 37]  # B,C,D,E,F,G,H,J,K,L,N,O,R,S,Z,AL

# Custom headers for KBIMS CSV files (mapped to selected columns)
CUSTOM_HEADERS = [
    "분류-대-공정",        # Column B
    "분류-대-공정",        # Column C
    "분류-중-재료",        # Column D
    "분류-중-재료",        # Column E
    "분류-소-객체유형",     # Column F
    "분류-소-객체유형",     # Column G
    "라이브러리-라이브러리 ID(코드)",  # Column H
    "라이브러리-파일명",     # Column J
    "라이브러리-확장자-rfa",  # Column K
    "라이브러리-확장자-rte",  # Column L
    "라이브러리-명칭",      # Column N
    "라이브러리-명칭(유형명)",  # Column O
    "조달청 표준공사코드-세부공종코드(전체)",  # Column R
    "조달청 표준공사코드-세부공종코드(추가)",  # Column S
    "관련규격-KBIMS 부위분류",  # Column Z
    "비고"              # Column AL
]


def convert_sheet_to_csv(excel_file_path, sheet_name, output_dir="data/csv", skiprows=None):
    """
    Convert a specific sheet from Excel to CSV format

    Args:
        excel_file_path (str): Path to the Excel file
        sheet_name (str): Name of the sheet to convert
        output_dir (str): Output directory for CSV files
        skiprows (int): Number of header rows to skip
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Read Excel sheet with specific column filtering and header skipping
        # First, try to read with all selected columns
        df = pd.read_excel(
            excel_file_path,
            sheet_name=sheet_name,
            skiprows=skiprows,
            # Read only selected columns: B,C,D,E,F,G,H,J,K,L,N,O,R,S,Z,AL
            usecols=SELECTED_COLUMNS,
            engine='openpyxl'
        )

        # Apply custom headers to selected columns
        if len(df.columns) == len(CUSTOM_HEADERS):
            df.columns = CUSTOM_HEADERS
        elif len(df.columns) <= len(CUSTOM_HEADERS):
            # Use available headers for the columns we have
            df.columns = CUSTOM_HEADERS[:len(df.columns)]
        else:
            # Fallback: use generic headers if unexpected column count
            print(
                f"⚠️ Warning: Expected {len(CUSTOM_HEADERS)} columns but got {len(df.columns)} for sheet '{sheet_name}'")
            df.columns = [f"Column_{i+1}" for i in range(len(df.columns))]

        # Generate output filename
        output_filename = f"KBIMS_{sheet_name}.csv"
        output_path = os.path.join(output_dir, output_filename)

        # Save to CSV with UTF-8 encoding for Korean text
        df.to_csv(output_path, index=False, encoding='utf-8')

        print(
            f"✅ Successfully converted sheet '{sheet_name}' to {output_path}")
        print(f"   Data shape: {df.shape[0]} rows × {df.shape[1]} columns")

    except Exception as e:
        print(f"❌ Error converting sheet '{sheet_name}': {str(e)}")


def list_sheets(excel_file_path):
    """
    List all sheet names in the Excel file

    Args:
        excel_file_path (str): Path to the Excel file

    Returns:
        list: List of sheet names
    """
    try:
        xl_file = pd.ExcelFile(excel_file_path, engine='openpyxl')
        return xl_file.sheet_names
    except Exception as e:
        print(f"Error reading Excel file: {str(e)}")
        return []


def main():
    excel_file = "data/xlsx/KBIMS.xlsx"

    # List all sheets first
    sheets = list_sheets(excel_file)
    if not sheets:
        print("❌ No sheets found or error reading Excel file")
        return

    print(f"📊 Found {len(sheets)} sheets: {sheets}")

    # Convert all sheets
    successful_conversions = 0
    failed_conversions = 0

    for sheet_name in sheets:
        print(f"\n🔄 Processing sheet: '{sheet_name}'")

        # Get skip rows configuration for this sheet
        # Default to 4 if not specified
        skip_rows = SHEET_SKIP_ROWS.get(sheet_name, 4)
        print(f"   Header rows to skip: {skip_rows}")

        try:
            convert_sheet_to_csv(excel_file, sheet_name, skiprows=skip_rows)
            successful_conversions += 1
        except Exception as e:
            print(f"❌ Failed to convert sheet '{sheet_name}': {str(e)}")
            failed_conversions += 1

    # Summary
    print("\n📈 Conversion Summary:")
    print(f"   ✅ Successful: {successful_conversions} sheets")
    print(f"   ❌ Failed: {failed_conversions} sheets")
    print("   📂 Output directory: data/csv/")

    if successful_conversions > 0:
        print("🎉 CSV conversion completed!")
    else:
        print("😞 No sheets were successfully converted.")


if __name__ == "__main__":
    main()
