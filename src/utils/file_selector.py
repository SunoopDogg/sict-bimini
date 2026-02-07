from pathlib import Path

DEFAULT_JSON_DIR = Path("data/json")


def select_json_file(directory: Path = DEFAULT_JSON_DIR) -> Path | None:
    """
    Display available JSON files and let user select one.

    Args:
        directory: Directory to search for JSON files

    Returns:
        Selected file path, or None if cancelled
    """
    if not directory.exists():
        print(f"Directory not found: {directory}")
        return None

    # Find all JSON files
    json_files = sorted(directory.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in: {directory}")
        return None

    # Display file list
    print(f"\nAvailable JSON files in '{directory}':")
    print("-" * 50)
    for i, file in enumerate(json_files, 1):
        size_kb = file.stat().st_size / 1024
        print(f"  [{i}] {file.name} ({size_kb:.1f} KB)")
    print("-" * 50)
    print("  [0] Cancel")
    print()

    # Get user selection
    while True:
        try:
            choice = input("Select file number: ").strip()

            if choice == "0" or choice.lower() == "q":
                print("Cancelled.")
                return None

            index = int(choice) - 1
            if 0 <= index < len(json_files):
                selected = json_files[index]
                print(f"Selected: {selected}")
                return selected
            else:
                print(f"Invalid selection. Enter 1-{len(json_files)} or 0 to cancel.")

        except ValueError:
            print("Please enter a valid number.")
        except KeyboardInterrupt:
            print("\nCancelled.")
            return None
