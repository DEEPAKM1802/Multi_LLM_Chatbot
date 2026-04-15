"""
Python Script: File Importer with Mapping (Scalable Design)
--------------------------------------------------------
This version introduces:
- Auto file type detection
- Mapping-based handler system (like real frameworks)
- Centralized error handling

This is a highly scalable and clean design.
"""

import csv
import json
import os


class FileImporter:
    "A reusable class to import CSV and JSON files"

    def __init__(self, file_path):
        self.file_path = file_path

        # Mapping file extensions to handler methods
        self._handlers = {
            ".csv": self._import_csv,
            ".json": self._import_json
        }

    # -----------------------------
    # Internal Validators
    # -----------------------------

    def _validate_exists(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

    def _get_file_extension(self):
        return os.path.splitext(self.file_path)[1].lower()

    # -----------------------------
    # Internal Import Logic
    # -----------------------------

    def _import_csv(self):
        data = []

        with open(self.file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)

            for row in reader:
                cleaned_row = {
                    k: v.strip() if isinstance(v, str) else v
                    for k, v in row.items()
                }
                data.append(cleaned_row)

        return data

    def _import_json(self):
        with open(self.file_path, mode='r', encoding='utf-8') as file:
            return json.load(file)

    # -----------------------------
    # Public Method (AUTO-DETECT + MAPPING)
    # -----------------------------

    def import_file(self):
        try:
            self._validate_exists()

            extension = self._get_file_extension()

            # Fetch handler from mapping
            handler = self._handlers.get(extension)

            if not handler:
                raise ValueError(f"Unsupported file type: {extension}")

            data = handler()

            print(f"File imported successfully: {self.file_path}")
            return data

        except FileNotFoundError as e:
            print(f"[ERROR] {e}")

        except json.JSONDecodeError as e:
            print(f"[ERROR] Invalid JSON format: {e}")

        except Exception as e:
            print(f"[ERROR] {e}")

        return None


# --------------------------------------------------
# Example Usage
# --------------------------------------------------

if __name__ == "__main__":

    importer = FileImporter("employees_realistic.csv")
    data = importer.import_file()

    if data:
        print(f"Total records: {len(data)}")
        print("Sample:", data[0])

    print("\n" + "-" * 50 + "\n")

    importer = FileImporter("transactions_realistic.json")
    data = importer.import_file()

    if data:
        print(f"Total records: {len(data)}")
        print("Sample:", data[0])


"""
🚀 Why this is powerful:
- Easily extendable → add '.xml': self._import_xml
- No if-else clutter → mapping handles routing
- Cleaner, scalable, industry-style design
"""
