import csv
import json
import os
import re
from datetime import datetime

# --------------------------------------------------
# File Importer
# --------------------------------------------------
class FileImporter:
    "A reusable class to import CSV and JSON files"
    def __init__(self):
        self.file_path = None
        self._handlers = {".csv": self._import_csv, ".json": self._import_json}

    def _validate_exists(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
    def _get_file_extension(self):
        return os.path.splitext(self.file_path)[1].lower()

    def _import_csv(self):
        try:
            data = []
            with open(self.file_path, mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                data = list(map(lambda row: {k: v.strip() if isinstance(v, str) else v for k, v in row.items()}, reader))
            return data
        except Exception as e:
            print(f"[ERROR] Could not import CSV file '{self.file_path}': {str(e)}")
            return None

    def _import_json(self):
        try:
            with open(self.file_path, mode='r', encoding='utf-8') as file:
                return json.load(file)
        except Exception as e:
            print(f"[ERROR] Could not import JSON file '{self.file_path}': {str(e)}")
            return None

    def import_file(self, file_path):
        self.file_path = file_path
        try:
            self._validate_exists()
            handler = self._handlers.get(self._get_file_extension())
            if not handler:
                raise ValueError(f"Unsupported format: {self._get_file_extension()}")
            data = handler()
            if data is not None:
                print(f"[INFO] Successfully imported: {self.file_path}")
            return data
        except Exception as e:
            print(f"[ERROR] Initialization failed for file '{self.file_path}': {e}")
            return None

# --------------------------------------------------
# Data Processor
# --------------------------------------------------
class DataProcessor:
    VALID_DEPARTMENTS = {'IT', 'Engineering', 'Admin', 'Support', 'Finance', 'HR', 'Sales', 'Marketing'}
    VALID_STATUSES = {'success', 'failed', 'unknown', 'pass', 'fail'}
    NAME_REGEX = re.compile(r'^[a-zA-Z\s]+$')

    def __init__(self, employees_data, transactions_data):
        self.employees = employees_data
        self.transactions = transactions_data
        self.processed_data = []

    def map_data(self):
        print("[INFO] Mapping data...")
        try:
            emp_lookup = {str(emp['user_id']): emp for emp in self.employees if 'user_id' in emp}
            
            def merge_record(trans):
                user_id = str(trans.get('user_id'))
                return {**emp_lookup.get(user_id, {}), **trans}
                
            self.processed_data = list(map(merge_record, self.transactions))
            return self.processed_data
        except Exception as e:
            print(f"[ERROR] Could not map employee and transaction data: {str(e)}")
            return self.transactions

    def remove_duplicates(self, data):
        print("[INFO] Removing duplicates...")
        try:
            seen = set()
            def is_unique(record):
                record_tuple = tuple(sorted((k, str(v)) for k, v in record.items()))
                if record_tuple not in seen:
                    seen.add(record_tuple)
                    return True
                return False
                
            return list(filter(is_unique, data))
        except Exception as e:
            print(f"[ERROR] Could not remove duplicates from data: {str(e)}")
            return data

    def handle_missing_values(self, data):
        print("[INFO] Handling missing values...")
        try:
            def clean_record(record):
                new_rec = record.copy()
                if not new_rec.get('department') or str(new_rec.get('department')).strip() == '':
                    new_rec['department'] = 'Unknown'
                if not new_rec.get('name') or str(new_rec.get('name')).strip() == '':
                    new_rec['name'] = 'Unknown'
                if new_rec.get('amount') is None:
                    new_rec['amount'] = 0.0
                return new_rec
                
            return list(map(clean_record, data))
        except Exception as e:
            print(f"[ERROR] Could not handle missing values: {str(e)}")
            return data

    def validate_fields(self, data):
        print("[INFO] Validating fields (Age, login_count, name, department, status)...")
        try:
            def validate_record(record):
                new_rec = record.copy()
                errors = []
                
                name = str(new_rec.get('name', ''))
                if not self.NAME_REGEX.match(name):
                    errors.append("Invalid Name Format")
                    new_rec['name'] = "Unknown"
                    
                dept = str(new_rec.get('department', ''))
                if dept not in self.VALID_DEPARTMENTS and dept != 'Unknown':
                    errors.append("Invalid Department")
                    new_rec['department'] = "Unknown"
                    
                status = str(new_rec.get('status', 'unknown')).lower()
                if status not in self.VALID_STATUSES:
                    errors.append("Invalid Status")
                    new_rec['status'] = "unknown"
                else:
                    new_rec['status'] = status
                    
                age_val = new_rec.get('Age')
                try:
                    age_int = int(age_val)
                    if not (18 <= age_int <= 100):
                        raise ValueError
                    new_rec['Age'] = age_int
                except (ValueError, TypeError):
                    errors.append("Invalid Age")
                    new_rec['Age'] = None
                    
                login_val = new_rec.get('login_count')
                try:
                    login_int = int(login_val)
                    if login_int < 0:
                        raise ValueError
                    new_rec['login_count'] = login_int
                except (ValueError, TypeError):
                    errors.append("Invalid Login Count")
                    new_rec['login_count'] = 0
                    
                new_rec['validation_errors'] = errors
                return new_rec
                
            return list(map(validate_record, data))
        except Exception as e:
            print(f"[ERROR] Could not validate specific data fields: {str(e)}")
            return data

    def correct_data_types(self, data):
        print("[INFO] Correcting data types...")
        try:
            def convert_types(record):
                new_rec = record.copy()
                if 'user_id' in new_rec and new_rec['user_id'] is not None:
                    try: new_rec['user_id'] = int(new_rec['user_id'])
                    except: pass
                if 'amount' in new_rec and new_rec['amount'] is not None:
                    try: new_rec['amount'] = float(new_rec['amount'])
                    except: new_rec['amount'] = 0.0
                return new_rec
                
            return list(map(convert_types, data))
        except Exception as e:
            print(f"[ERROR] Could not enforce data types on the dataset: {str(e)}")
            return data

    def validate_and_format_dates(self, data):
        print("[INFO] Validating and formatting dates...")
        try:
            date_fields = ['last_login', 'date_of_transaction']
            
            def format_dates(record):
                new_rec = record.copy()
                for field in date_fields:
                    val = str(new_rec.get(field, '')).strip()
                    if val in ('invalid-date', 'None', ''):
                        new_rec[field] = None
                    else:
                        try:
                            datetime.strptime(val, '%Y-%m-%d')
                            new_rec[field] = val
                        except ValueError:
                            new_rec[field] = None
                return new_rec
                
            return list(map(format_dates, data))
        except Exception as e:
            print(f"[ERROR] Could not validate and format dates: {str(e)}")
            return data

    def process(self):
        print("[INFO] Starting data processing...")
        data = self.map_data()
        data = self.remove_duplicates(data)
        data = self.handle_missing_values(data)
        data = self.validate_fields(data)
        data = self.correct_data_types(data)
        data = self.validate_and_format_dates(data)
        
        self.processed_data = data
        print(f"[INFO] Data processing complete. Total records: {len(self.processed_data)}")
        return self.processed_data

# --------------------------------------------------
# Insights Generator
# --------------------------------------------------
class InsightsGenerator:
    def __init__(self, data):
        if not isinstance(data, list):
            raise ValueError("Input data must be a list of dictionaries.")
        self.data = data
        self.insights = {}

    def total_revenue_by_department(self):
        print("[INFO] Calculating total revenue by department...")
        try:
            success_records = list(filter(lambda r: r.get('status') in ('success', 'pass'), self.data))
            revenue_by_dept = {}
            for r in success_records:
                dept = r.get('department', 'Unknown')
                revenue_by_dept[dept] = revenue_by_dept.get(dept, 0.0) + r.get('amount', 0.0)
                
            self.insights['total_revenue_by_department'] = {k: round(v, 2) for k, v in revenue_by_dept.items()}
        except Exception as e:
            print(f"[ERROR] Could not calculate the total revenue by department: {str(e)}")

    def top_performing_employees(self, top_n=3):
        print(f"[INFO] Finding top {top_n} performing employees...")
        try:
            success_records = list(filter(lambda r: r.get('status') in ('success', 'pass'), self.data))
            
            emp_revenue = {}
            for r in success_records:
                name = r.get('name', 'Unknown')
                emp_revenue[name] = emp_revenue.get(name, 0.0) + r.get('amount', 0.0)
                
            sorted_emps = sorted(emp_revenue.items(), key=lambda item: item[1], reverse=True)
            self.insights['top_performing_employees'] = list(map(
                lambda x: {"name": x[0], "total_amount": round(x[1], 2)}, sorted_emps[:top_n]
            ))
        except Exception as e:
            print(f"[ERROR] Could not find the top performing employees: {str(e)}")

    def transaction_status_breakdown(self):
        print("[INFO] Generating transaction status breakdown...")
        try:
            statuses = list(map(lambda r: r.get('status', 'unknown'), self.data))
            status_counts = {}
            for s in statuses:
                status_counts[s] = status_counts.get(s, 0) + 1
                
            self.insights['transaction_status_breakdown'] = status_counts
        except Exception as e:
            print(f"[ERROR] Could not generate the transaction status breakdown: {str(e)}")

    def data_quality_report(self):
        print("[INFO] Generating data quality report...")
        try:
            error_records = list(filter(lambda r: len(r.get('validation_errors', [])) > 0, self.data))
            
            error_counts = {}
            for err_list in map(lambda r: r.get('validation_errors', []), error_records):
                for err in err_list:
                    error_counts[err] = error_counts.get(err, 0) + 1
                    
            self.insights['data_quality_report'] = {
                "total_records_with_errors": len(error_records),
                "error_breakdown": error_counts
            }
        except Exception as e:
            print(f"[ERROR] Could not generate the data quality report: {str(e)}")

    def generate_report(self):
        self.total_revenue_by_department()
        self.top_performing_employees()
        self.transaction_status_breakdown()
        self.data_quality_report()
        return self.insights

    def export_to_json(self, output_path="insights_report.json"):
        if not self.insights:
            print("[WARNING] No insights to export. Generating report first.")
            self.generate_report()
            
        print(f"[INFO] Exporting insights to {output_path}...")
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.insights, f, indent=4)
            print(f"[INFO] Successfully exported insights to {output_path}")
        except Exception as e:
            print(f"[ERROR] Could not export insights to JSON file: {str(e)}")

# --------------------------------------------------
# Main Execution
# --------------------------------------------------
if __name__ == "__main__":
    csv_file = "employees_realistic.csv"
    json_file = "transactions_realistic.json"
    
    print("-" * 50)
    print("Loading Data...")
    importer = FileImporter()
    employees_data = importer.import_file(csv_file)
    transactions_data = importer.import_file(json_file)
    
    if employees_data is not None and transactions_data is not None:
        print("-" * 50)
        print("Processing Data...")
        processor = DataProcessor(employees_data, transactions_data)
        cleaned_data = processor.process()
        
        if cleaned_data:
            print("-" * 50)
            print("Generating Insights...")
            generator = InsightsGenerator(cleaned_data)
            generator.generate_report()
            generator.export_to_json("insights_report.json")
            print("-" * 50)
            print("Consolidated pipeline completed gracefully.")
    else:
        print("[ERROR] Pipeline halted due to missing or corrupt data.")
