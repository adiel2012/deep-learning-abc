
import json
import glob
import sys

def validate_notebooks():
    notebooks = glob.glob("*.ipynb")
    print(f"Found {len(notebooks)} notebooks.")
    failed = []
    for nb_file in notebooks:
        try:
            with open(nb_file, 'r', encoding='utf-8') as f:
                nb = json.load(f)
            # Basic check for cells and metadata
            if 'cells' not in nb or 'metadata' not in nb:
                print(f"❌ {nb_file}: Missing 'cells' or 'metadata'")
                failed.append(nb_file)
            else:
                print(f"✅ {nb_file}: Valid JSON structure")
        except Exception as e:
            print(f"❌ {nb_file}: JSON Error - {e}")
            failed.append(nb_file)
    
    if failed:
        print(f"\nFailed validation: {failed}")
        sys.exit(1)
    else:
        print("\nAll notebooks validated successfully.")

if __name__ == "__main__":
    validate_notebooks()
