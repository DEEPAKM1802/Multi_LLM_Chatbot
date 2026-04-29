import csv
import json
import os
import re
from datetime import datetime
from functools import reduce
from collections import Counter
from dataclasses import dataclass, fields
from typing import Type, Optional, List, Dict, Any

# =========================================================
# SOLID Data Analytics Pipeline
# =========================================================

@dataclass
class EmployeeSchema:
    user_id: str
    department: str
    name: str
    last_login: str
    Age: int
    login_count: int

@dataclass
class TransactionSchema:
    user_id: str
    amount: float
    status: str
    date_of_transaction: str

class DataLoader:
    """SRP: Handles reading data from different file formats."""

    @staticmethod
    def _validate_schema(row: dict, schema: Optional[Type]) -> None:
        """Validates that all required fields in the schema are present in the row."""
        if schema is None:
            return
            
        required_fields = {f.name for f in fields(schema)}
        missing_fields = required_fields - set(row.keys())
        
        if missing_fields:
            raise ValueError(f"Missing required columns: {', '.join(missing_fields)}")

    @classmethod
    def _load_csv(cls, file_path: str, schema: Optional[Type] = None) -> list:
        """Loads and parses a CSV file."""
        data = []
        try:
            with open(file_path, mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    cleaned_row = {k: v.strip() if isinstance(v, str) else v for k, v in row.items()}
                    cls._validate_schema(cleaned_row, schema)
                    data.append(cleaned_row)
        except csv.Error as e:
            raise csv.Error(f"CSV parsing error in {file_path}: {e}")
        except IOError as e:
            raise IOError(f"I/O error reading {file_path}: {e}")
        return data

    @classmethod
    def _load_json(cls, file_path: str, schema: Optional[Type] = None) -> list:
        """Loads and parses a JSON file."""
        try:
            with open(file_path, mode='r', encoding='utf-8') as file:
                data = json.load(file)
                
                if not isinstance(data, list):
                    raise ValueError(f"JSON data in {file_path} must be a list of dictionaries.")
                    
                for row in data:
                    if not isinstance(row, dict):
                        raise ValueError("Each item in the JSON list must be a dictionary.")
                    cls._validate_schema(row, schema)
                return data
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON decoding error in {file_path}: {e.msg}") from e
        except IOError as e:
            raise IOError(f"I/O error reading {file_path}: {e}")

    @classmethod
    def load(cls, file_path: str, schema: Optional[Type] = None) -> list:
        """Main entry point to load data based on file extension."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.csv':
            return cls._load_csv(file_path, schema)
        elif ext == '.json':
            return cls._load_json(file_path, schema)
        else:
            raise ValueError(f"Unsupported format: {ext}")


class DataCleaner:
    """SRP: Cleans, merges, and validates raw records using functional pipelines."""
    _VALID_DEPARTMENTS = frozenset({'IT', 'Engineering', 'Admin', 'Support', 'Finance', 'HR', 'Sales', 'Marketing'})
    _VALID_STATUSES = frozenset({'success', 'failed'})
    _NAME_REGEX = re.compile(r'^[a-zA-Z\s]+$')

    def __init__(self, employees, transactions):
        self.employees = employees
        self.transactions = transactions

    def _merge_data(self):
        """Merges employee and transaction data using O(1) lookup, assigning a transaction ID."""
        emp_lookup = {str(e.get('user_id')): e for e in self.employees if e.get('user_id') is not None}
        for i, t in enumerate(self.transactions):
            merged = {**emp_lookup.get(str(t.get('user_id')), {}), **t}
            # if 'transaction_id' not in merged:
            #     merged['transaction_id'] = f"TXN_{i+1:05d}"
            yield merged

    def _deduplicate_by_row(self, data_iter):
        """Filters out identical duplicate rows using a closure."""
        seen = set()
        seen_add = seen.add

        for rec in data_iter:
            rtup = tuple((k, str(v)) for k, v in rec.items())

            if rtup in seen:
                continue

            seen_add(rtup)
            yield rec

    def _handle_missing_values(self, record):
        """Replaces missing values with None and logs them."""
        rec = record.copy()
        errors = rec.get('validation_errors', [])
        missing_indicators = {'', 'none', 'nan', 'null'}
        try:
            for k, v in rec.items():
                if k == 'validation_errors':
                    continue
                if v is None or str(v).strip().lower() in missing_indicators:
                    rec[k] = None
                    errors.append(f"Missing value for {k}")
        except Exception as e:
            errors.append(f"Unexpected error handling missing values: {e}")
                
        rec['validation_errors'] = errors
        return rec

    def _validate_user_id(self, record):
        """Ensures user_id is properly formatted."""
        rec = record.copy()
        try:
            if rec.get('user_id') is not None:
                rec['user_id'] = str(rec['user_id']).strip()
            else:
                rec.setdefault('validation_errors', []).append("Missing User ID")
                rec['user_id'] = None
        except Exception as e:
            rec.setdefault('validation_errors', []).append(f"Unexpected error parsing user_id: {e}")
            rec['user_id'] = None
        return rec

    def _validate_name(self, record):
        """Validates that the name matches required regex."""
        rec = record.copy()
        errors = rec.get('validation_errors', [])
        try:
            name = rec.get('name')
            if name is not None:
                name_str = str(name).strip()
                if self._NAME_REGEX.match(name_str):
                    rec['name'] = name_str
                else:
                    errors.append(f"Invalid Name: {name}")
                    rec['name'] = None
            else:
                errors.append("Missing Name")
                rec['name'] = None
        except Exception as e:
            errors.append(f"Unexpected error validating name: {e}")
            rec['name'] = None
        rec['validation_errors'] = errors
        return rec

    def _validate_department(self, record):
        """Validates department against listed valid departments."""
        rec = record.copy()
        errors = rec.get('validation_errors', [])
        try:
            dept = rec.get('department')
            if dept is not None:
                dept_str = str(dept).strip()
                if dept_str in self._VALID_DEPARTMENTS:
                    rec['department'] = dept_str
                else:
                    errors.append(f"Invalid Dept: {dept}")
                    rec['department'] = None
            else:
                errors.append("Missing Dept")
                rec['department'] = None
        except Exception as e:
            errors.append(f"Unexpected error validating department: {e}")
            rec['department'] = None
        rec['validation_errors'] = errors
        return rec

    def _validate_status(self, record):
        """Validates status against listed valid statuses."""
        rec = record.copy()
        errors = rec.get('validation_errors', [])
        try:
            status = rec.get('status')
            if status is not None:
                status_str = str(status).strip().lower()
                if status_str in self._VALID_STATUSES:
                    rec['status'] = status_str
                else:
                    errors.append(f"Invalid Status: {status}")
                    rec['status'] = None
            else:
                errors.append("Missing Status")
                rec['status'] = None
        except Exception as e:
            errors.append(f"Unexpected error validating status: {e}")
            rec['status'] = None
        rec['validation_errors'] = errors
        return rec

    def _validate_amount(self, record):
        """Validates amount is > 0 and safely casts to float."""
        rec = record.copy()
        errors = rec.get('validation_errors', [])
        try:
            amount = rec.get('amount')
            if amount is not None:
                try:
                    val = float(amount)
                    if val <= 0:
                        errors.append(f"Invalid Amount (<=0): {val}")
                        rec['amount'] = None
                    else:
                        rec['amount'] = val
                except (ValueError, TypeError):
                    errors.append(f"Invalid Amount format: {amount}")
                    rec['amount'] = None
            else:
                errors.append("Missing Amount")
                rec['amount'] = None
        except Exception as e:
            errors.append(f"Unexpected error validating amount: {e}")
            rec['amount'] = None
        rec['validation_errors'] = errors
        return rec

    def _validate_login_count(self, record):
        """Validates login_count is >= 0 and safely casts to int."""
        rec = record.copy()
        errors = rec.get('validation_errors', [])
        try:
            login_count = rec.get('login_count')
            if login_count is not None:
                try:
                    val = int(login_count)
                    if val < 0:
                        errors.append(f"Invalid Login Count (<0): {val}")
                        rec['login_count'] = None
                    else:
                        rec['login_count'] = val
                except (ValueError, TypeError):
                    errors.append(f"Invalid Login Count format: {login_count}")
                    rec['login_count'] = None
            else:
                errors.append("Missing Login Count")
                rec['login_count'] = None
        except Exception as e:
            errors.append(f"Unexpected error validating login_count: {e}")
            rec['login_count'] = None
        rec['validation_errors'] = errors
        return rec

    def _validate_age(self, record):
        """Validates age is between 18 and 100."""
        rec = record.copy()
        errors = rec.get('validation_errors', [])
        try:
            age = rec.get('Age')
            if age is not None:
                try:
                    val = int(age)
                    if not (18 <= val <= 100):
                        errors.append(f"Invalid Age ({val} not in 18-100)")
                        rec['Age'] = None
                    else:
                        rec['Age'] = val
                except (ValueError, TypeError):
                    errors.append(f"Invalid Age format: {age}")
                    rec['Age'] = None
            else:
                errors.append("Missing Age")
                rec['Age'] = None
        except Exception as e:
            errors.append(f"Unexpected error validating age: {e}")
            rec['Age'] = None
        rec['validation_errors'] = errors
        return rec

    def _validate_logic(self, record):
        """Checks business logic: e.g. login_count 0 means no transaction."""
        rec = record.copy()
        errors = rec.get('validation_errors', [])
        try:
            login_count = rec.get('login_count')
            amount = rec.get('amount')
            
            if login_count == 0 and amount is not None and amount > 0:
                errors.append("Logic Error: Transaction exists but login count is 0")
                rec['amount'] = None
        except Exception as e:
            errors.append(f"Unexpected error validating logic: {e}")
            
        rec['validation_errors'] = errors
        return rec

    def _convert_dates(self, record):
        """Safely parses date fields."""
        rec = record.copy()
        errors = rec.get('validation_errors', [])
        try:
            for df in ['last_login', 'date_of_transaction']:
                val = rec.get(df)
                if val is not None:
                    try:
                        datetime.strptime(str(val).strip(), '%Y-%m-%d')
                        rec[df] = str(val).strip()
                    except ValueError:
                        errors.append(f"Invalid Date: {val}")
                        rec[df] = None
        except Exception as e:
            errors.append(f"Unexpected error validating dates: {e}")
            
        rec['validation_errors'] = errors
        return rec

    def process(self):
        """Orchestrates data cleaning pipeline using map and filter."""
        mapped_iter = self._merge_data()
        
        # Need a materialized list for 'mapped' as it is returned directly
        mapped_list = list(mapped_iter)
        
        # Build functional pipeline
        pipeline = self._deduplicate_by_row(mapped_list)
        pipeline = map(self._handle_missing_values, pipeline)
        pipeline = map(self._validate_user_id, pipeline)
        pipeline = map(self._validate_name, pipeline)
        pipeline = map(self._validate_department, pipeline)
        pipeline = map(self._validate_status, pipeline)
        pipeline = map(self._validate_amount, pipeline)
        pipeline = map(self._validate_login_count, pipeline)
        pipeline = map(self._validate_age, pipeline)
        pipeline = map(self._validate_logic, pipeline)
        pipeline = map(self._convert_dates, pipeline)
        
        cleaned_list = list(pipeline)
        
        return mapped_list, cleaned_list


class FeatureEngineer:
    """SRP: Builds user features and profiles with transaction nesting."""
    def __init__(self, cleaned_data, employees):
        self.cleaned_data = cleaned_data
        self.emp_ids = set(str(e.get('user_id', '')).strip() for e in employees if e.get('user_id'))
        
        valid_amts = sorted([r.get('amount') for r in cleaned_data if isinstance(r.get('amount'), (int, float))])
        n_amts = len(valid_amts)
        self.amt_95th = valid_amts[int(n_amts*0.95)] if n_amts else 0
        self.amt_5th = valid_amts[int(n_amts*0.05)] if n_amts else 0
        
        if n_amts:
            q1_amt, q3_amt = valid_amts[n_amts//4], valid_amts[(3*n_amts)//4]
            iqr_amt = q3_amt - q1_amt
            self.amt_upper = q3_amt + 1.5 * iqr_amt
            self.amt_lower = q1_amt - 1.5 * iqr_amt
        else:
            self.amt_upper = self.amt_lower = 0
            
        valid_logins = sorted([r.get('login_count') for r in cleaned_data if isinstance(r.get('login_count'), (int, float))])
        n_logs = len(valid_logins)
        self.log_95th = valid_logins[int(n_logs*0.95)] if n_logs else 0
        self.log_5th = valid_logins[int(n_logs*0.05)] if n_logs else 0
        
        if n_logs:
            q1_log, q3_log = valid_logins[n_logs//4], valid_logins[(3*n_logs)//4]
            iqr_log = q3_log - q1_log
            self.log_upper = q3_log + 1.5 * iqr_log
            self.log_lower = q1_log - 1.5 * iqr_log
        else:
            self.log_upper = self.log_lower = 0

    def build_features(self):
        users = {}
        for r in self.cleaned_data:
            uid = str(r.get('user_id', 'Unknown'))
            if uid not in users:
                users[uid] = {
                    "user_id": uid,
                    "name": r.get('name'),
                    "department": r.get('department'),
                    "age": r.get('Age'),
                    "last_login": r.get('last_login'),
                    "login_count": r.get('login_count'),
                    "transactions": []
                }
            
            txn = {
                "transaction_id": r.get('transaction_id'),
                "amount": r.get('amount'),
                "status": r.get('status'),
                "date_of_transaction": r.get('date_of_transaction'),
                "validation_errors": r.get('validation_errors', [])
            }
            users[uid]["transactions"].append(txn)
            
        features = []
        for uid, u in users.items():
            log_c = u.get("login_count")
            
            high_login = 1 if log_c is not None and log_c > self.log_upper else 0
            
            has_high_txn = any(t['amount'] > self.amt_upper for t in u['transactions'] if t['amount'] is not None)
            high_transaction = 1 if has_high_txn else 0
            
            has_95th_txn = any(t['amount'] >= self.amt_95th for t in u['transactions'] if t['amount'] is not None)
            high_user_activity = 1 if (log_c is not None and log_c >= self.log_95th) or has_95th_txn else 0
            
            has_5th_txn = any(t['amount'] <= self.amt_5th for t in u['transactions'] if t['amount'] is not None)
            low_activity_flag = 1 if (log_c is not None and log_c <= self.log_5th) or has_5th_txn else 0
            
            suspicious_reasons = []
            if uid not in self.emp_ids:
                suspicious_reasons.append("User not in CSV")
            if has_high_txn or any(t['amount'] < self.amt_lower for t in u['transactions'] if t['amount'] is not None):
                suspicious_reasons.append("Outlier in amount")
            if any(t['amount'] <= 0 for t in u['transactions'] if t['amount'] is not None):
                suspicious_reasons.append("Negative or 0 amount transaction")
            if high_login:
                suspicious_reasons.append("Outlier in login count")
                
            suspicious_flag = 1 if suspicious_reasons else 0
            
            anomaly_reasons = set()
            for t in u['transactions']:
                for err in t['validation_errors']:
                    anomaly_reasons.add(err)
            
            anomaly_flag = 1 if anomaly_reasons else 0
            
            u["high_login"] = high_login
            u["high_transaction"] = high_transaction
            u["high_user_activity"] = high_user_activity
            u["low_activity_flag"] = low_activity_flag
            u["suspicious_flag"] = suspicious_flag
            u["suspicious_reason"] = ", ".join(suspicious_reasons) if suspicious_reasons else None
            u["anomaly_flag"] = anomaly_flag
            u["anomaly_reason"] = ", ".join(anomaly_reasons) if anomaly_reasons else None
            
            features.append(u)
            
        return features


class MetricsAggregator:
    """SRP: Computes standard EDA and business metrics."""

    @staticmethod
    def _pearson_correlation(x, y):
        try:
            n = len(x)
            if n == 0: return 0.0
            sum_x = sum(x)
            sum_y = sum(y)
            sum_x_sq = sum(xi*xi for xi in x)
            sum_y_sq = sum(yi*yi for yi in y)
            psum = sum(xi*yi for xi, yi in zip(x, y))
            num = psum - (sum_x * sum_y / n)
            den = ((sum_x_sq - pow(sum_x, 2) / n) * (sum_y_sq - pow(sum_y, 2) / n)) ** 0.5
            if den == 0: return 0.0
            return num / den
        except Exception:
            return 0.0

    @staticmethod
    def _univariate_numeric(data, column):
        try:
            vals = sorted([r[column] for r in data if r.get(column) is not None])
            if not vals:
                return {"min": 0, "max": 0, "mean": 0, "median": 0, "std_dev": 0, "25%": 0, "75%": 0, "outliers_count": 0}
            
            n = len(vals)
            mean = sum(vals) / n
            median = vals[n//2] if n % 2 != 0 else (vals[n//2 - 1] + vals[n//2]) / 2.0
            variance = sum((x - mean) ** 2 for x in vals) / n
            std_dev = variance ** 0.5
            q1 = vals[n//4]
            q3 = vals[(3*n)//4]
            iqr = q3 - q1
            outliers = sum(1 for x in vals if x < (q1 - 1.5*iqr) or x > (q3 + 1.5*iqr))
            
            return {
                "min": min(vals), "max": max(vals), "mean": round(mean, 2),
                "median": round(median, 2), "std_dev": round(std_dev, 2),
                "25%": round(q1, 2), "75%": round(q3, 2), "outliers_count": outliers
            }
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def _univariate_categorical(data, column):
        try:
            vals = [str(r.get(column, 'Unknown')).strip() for r in data]
            counts = dict(Counter(vals))
            total = len(vals) or 1
            percents = {k: round((v / total) * 100, 2) for k, v in counts.items()}
            return {"counts": counts, "percents": percents}
        except Exception as e:
            return {"error": str(e)}

    @classmethod
    def _age_stats(cls, data):
        try:
            buckets = ["18-30", "31-40", "41-50", "51-60", "60+", "Unknown"]
            age_groups = {k: {"users": set(), "total_logins": 0, "amounts": [], "success": 0, "failed": 0, "unknown_status": 0, "missing": 0} for k in buckets}
            
            user_logins = {}
            
            for r in data:
                age = r.get('Age')
                if age is None: b_key = "Unknown"
                elif age <= 30: b_key = "18-30"
                elif age <= 40: b_key = "31-40"
                elif age <= 50: b_key = "41-50"
                elif age <= 60: b_key = "51-60"
                else: b_key = "60+"
                
                uid = str(r.get('user_id', '')).strip()
                if uid and uid != '-1' and uid != 'None':
                    if uid not in user_logins:
                        user_logins[uid] = {"bucket": b_key, "login": 0}
                    user_logins[uid]["bucket"] = b_key
                    if r.get('login_count') is not None:
                        user_logins[uid]["login"] = max(user_logins[uid]["login"], r['login_count'])
                        
                st = str(r.get('status')).strip().lower()
                if st == 'success': age_groups[b_key]["success"] += 1
                elif st == 'failed': age_groups[b_key]["failed"] += 1
                else: age_groups[b_key]["unknown_status"] += 1
                
                if r.get('amount') is not None:
                    age_groups[b_key]["amounts"].append(r['amount'])
                    
                has_missing = any(v is None and k != 'validation_errors' for k, v in r.items())
                if has_missing:
                    age_groups[b_key]["missing"] += 1
                
            for uid, info in user_logins.items():
                b_key = info["bucket"]
                age_groups[b_key]["users"].add(uid)
                age_groups[b_key]["total_logins"] += info["login"]
                
            stats = {}
            for k, v in age_groups.items():
                if v["users"] or v["amounts"] or v["missing"] > 0 or v["success"] > 0 or v["failed"] > 0 or v["unknown_status"] > 0:
                    avg_amt = sum(v["amounts"])/len(v["amounts"]) if v["amounts"] else 0
                    user_c = len(v["users"])
                    avg_logins = v["total_logins"] / user_c if user_c > 0 else 0
                    stats[k] = {
                        "user_count": user_c,
                        "avg_logins": round(avg_logins, 2),
                        "avg_amount": round(avg_amt, 2),
                        "success": v["success"],
                        "failed": v["failed"],
                        "unknown_status": v["unknown_status"],
                        "missing": v["missing"]
                    }
                    
            valid = [r for r in data if r.get('Age') is not None and r.get('amount') is not None]
            ages = [r['Age'] for r in valid]
            amounts = [r['amount'] for r in valid]
            corr = cls._pearson_correlation(ages, amounts)
            
            return {"correlation": round(corr, 4), "buckets": stats}
        except Exception as e:
            return {"error": str(e)}

    @classmethod
    def _correlation_matrix(cls, data):
        try:
            cols = ["Age", "login_count", "amount"]
            matrix = {c: {} for c in cols}
            for c1 in cols:
                for c2 in cols:
                    if c1 == c2:
                        matrix[c1][c2] = 1.0
                    elif c1 > c2:
                        valid = [r for r in data if r.get(c1) is not None and r.get(c2) is not None]
                        v1 = [r[c1] for r in valid]
                        v2 = [r[c2] for r in valid]
                        corr = cls._pearson_correlation(v1, v2)
                        matrix[c1][c2] = round(corr, 4)
                        matrix[c2][c1] = round(corr, 4)
            return matrix
        except Exception as e:
            return {"error": str(e)}

    @classmethod
    def _bivariate_login_vs_amount(cls, data):
        try:
            valid = [r for r in data if r.get('login_count') is not None and r.get('amount') is not None]
            logins = [r['login_count'] for r in valid]
            amounts = [r['amount'] for r in valid]
            corr = cls._pearson_correlation(logins, amounts)
            return {"correlation": round(corr, 4)}
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def _bivariate_status_vs_login(data):
        try:
            valid = [r for r in data if r.get('login_count') is not None]
            status_logins = {}
            for r in valid:
                status = str(r.get('status', 'unknown')).strip()
                status_logins.setdefault(status, []).append(r['login_count'])
            return {k: round(sum(v)/len(v), 2) for k, v in status_logins.items()}
        except Exception as e:
            return {"error": str(e)}
            
    @staticmethod
    def _overall_stats(raw_data, cleaned_data, employees):
        try:
            emp_users = set(str(e.get('user_id', '')).strip() for e in employees if str(e.get('user_id', '')).strip())
            
            trans_users_not_in_csv = set()
            problematic_users = set()
            missing_user_id_records = 0
            
            for r in cleaned_data:
                uid = r.get('user_id')
                if uid is None or str(uid).strip() in ('-1', 'None', '', 'unknown'):
                    missing_user_id_records += 1
                else:
                    uid_str = str(uid).strip()
                    if uid_str not in emp_users:
                        trans_users_not_in_csv.add(uid_str)
                    
                    if r.get('validation_errors'):
                        problematic_users.add(uid_str)
                        
            total_problematic = len(problematic_users) + missing_user_id_records
            
            cols = set().union(*(r.keys() for r in raw_data))
            missing_per_col = {c: sum(1 for r in cleaned_data if r.get(c) is None and c != 'validation_errors') for c in cols}
            
            missing_per_dept = {}
            for r in cleaned_data:
                dept = r.get('department')
                dept_key = str(dept).strip() if dept is not None else "Unknown"
                has_missing = any(v is None and k != 'validation_errors' for k, v in r.items())
                if has_missing:
                    missing_per_dept[dept_key] = missing_per_dept.get(dept_key, 0) + 1

            return {
                "total_users_csv": len(emp_users),
                "total_users_json_not_csv": len(trans_users_not_in_csv),
                "problematic_users": len(problematic_users),
                "missing_user_id_records": missing_user_id_records,
                "total_problematic": total_problematic,
                "missing_per_col": missing_per_col,
                "missing_per_dept": missing_per_dept
            }
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def _department_stats(data):
        try:
            user_logins = {}
            dept_details = {}
            
            for r in data:
                dept = str(r.get('department', 'Unknown')).strip()
                uid = str(r.get('user_id', '')).strip()
                
                if dept not in dept_details:
                    dept_details[dept] = {
                        "ages": [], 
                        "status_counts": {"success": 0, "failed": 0, "unknown": 0}
                    }
                
                if r.get('Age') is not None:
                    dept_details[dept]["ages"].append(r['Age'])
                
                st = str(r.get('status', 'unknown')).strip().lower()
                if st in dept_details[dept]["status_counts"]:
                    dept_details[dept]["status_counts"][st] += 1
                else:
                    dept_details[dept]["status_counts"]["unknown"] += 1
                
                if uid and uid != '-1':
                    if uid not in user_logins: 
                        user_logins[uid] = {"dept": dept, "login": 0, "amount": []}
                    user_logins[uid]["dept"] = dept
                    if r.get('login_count') is not None:
                        user_logins[uid]["login"] = max(user_logins[uid]["login"], r['login_count'])
                    if r.get('amount') is not None:
                        user_logins[uid]["amount"].append(r['amount'])
            
            final_dept = {}
            for uid, info in user_logins.items():
                d = info["dept"]
                if d not in final_dept: final_dept[d] = {"user_count": 0, "total_logins": 0, "amounts": []}
                final_dept[d]["user_count"] += 1
                final_dept[d]["total_logins"] += info["login"]
                final_dept[d]["amounts"].extend(info["amount"])
                
            stats = {}
            all_depts = set(final_dept.keys()).union(set(dept_details.keys()))
            for d in all_depts:
                v = final_dept.get(d, {"user_count": 0, "total_logins": 0, "amounts": []})
                avg_amt = sum(v["amounts"])/len(v["amounts"]) if v["amounts"] else 0
                
                ages = dept_details.get(d, {}).get("ages", [])
                avg_age = sum(ages)/len(ages) if ages else 0
                
                statuses = dept_details.get(d, {}).get("status_counts", {"success": 0, "failed": 0, "unknown": 0})
                
                stats[d] = {
                    "user_count": v["user_count"],
                    "total_logins": v["total_logins"],
                    "avg_amount": round(avg_amt, 2),
                    "avg_age": round(avg_age, 2),
                    "success": statuses["success"],
                    "failed": statuses["failed"],
                    "unknown_status": statuses["unknown"]
                }
                
            most_active_dept = max(stats.items(), key=lambda x: x[1]["total_logins"]) if stats else ("N/A", {"total_logins": 0})
            highest_trans_dept = max(stats.items(), key=lambda x: x[1]["avg_amount"]) if stats else ("N/A", {"avg_amount": 0})
            
            top_by_login = sorted(user_logins.items(), key=lambda x: x[1]["login"], reverse=True)[:5]
            top_by_amount = sorted(user_logins.items(), key=lambda x: sum(x[1]["amount"]), reverse=True)[:5]

            return {
                "department_stats": stats,
                "highest_user_activity_dept": most_active_dept[0],
                "highest_transaction_dept": highest_trans_dept[0],
                "most_active_users_by_login": {k: v["login"] for k, v in top_by_login},
                "top_users_by_spend": {k: round(sum(v["amount"]), 2) for k, v in top_by_amount}
            }
        except Exception as e:
            return {"error": str(e)}

    @classmethod
    def calculate_eda(cls, raw_data, cleaned_data, employees):
        try:
            return {
                "overall": cls._overall_stats(raw_data, cleaned_data, employees),
                "univariate_numeric": {
                    "Age": cls._univariate_numeric(cleaned_data, "Age"),
                    "login_count": cls._univariate_numeric(cleaned_data, "login_count"),
                    "amount": cls._univariate_numeric(cleaned_data, "amount")
                },
                "univariate_categorical": {
                    "department": cls._univariate_categorical(cleaned_data, "department"),
                    "status": cls._univariate_categorical(cleaned_data, "status")
                },
                "bivariate": {
                    "age_stats": cls._age_stats(cleaned_data),
                    "status_vs_avg_login": cls._bivariate_status_vs_login(cleaned_data),
                    "correlation_matrix": cls._correlation_matrix(cleaned_data)
                },
                "department_and_user_aggregations": cls._department_stats(cleaned_data)
            }
        except Exception as e:
            return {"error": f"Failed to calculate EDA: {e}"}


class ReportWriter:
    """SRP: Handles all file system writing."""
    @staticmethod
    def write_analysis_summary(eda):
        with open("analysis_summary.txt", "w", encoding="utf-8") as f:
            f.write("=== EXPLORATORY DATA ANALYSIS (EDA) ===\n\n")
            if "error" in eda:
                f.write(f"Error computing EDA: {eda['error']}\n")
                return

            ov = eda.get("overall", {})
            missing_per_col = ov.get('missing_per_col', {})
            missing_per_dept = ov.get('missing_per_dept', {})
            agg = eda.get("department_and_user_aggregations", {})
            bi = eda.get("bivariate", {})

            f.write("--- OVERALL STATS ---\n")
            f.write(f"Total Users (in CSV): {ov.get('total_users_csv', 0)}\n")
            f.write(f"Total Users (in JSON only, missing in CSV): {ov.get('total_users_json_not_csv', 0)}\n\n")
            f.write("Problematic Breakdown:\n")
            f.write(f"  - Users with >=1 formatting error: {ov.get('problematic_users', 0)}\n")
            f.write(f"  - Transactions missing User ID completely: {ov.get('missing_user_id_records', 0)}\n")
            f.write(f"  - Total Problematic Metric: {ov.get('total_problematic', 0)}\n\n")
            
            f.write("Missing Values for Other Columns:\n")
            for col, cnt in missing_per_col.items():
                if cnt > 0 and col.lower() not in ['age', 'login_count', 'amount', 'department', 'status']:
                    f.write(f"  - {col}: {cnt}\n")
                    
            f.write("\n--- DEPARTMENT & USER AGGREGATIONS ---\n")
            f.write(f"Highest User Activity (Logins) Dept: {agg.get('highest_user_activity_dept')}\n")
            f.write(f"Highest Avg Transaction Dept: {agg.get('highest_transaction_dept')}\n")
            
            f.write("\nMost Active Users (by Logins):\n")
            for u, l in agg.get("most_active_users_by_login", {}).items():
                f.write(f"  - User {u}: {l} logins\n")
                
            f.write("\nTop Users (by Total Spend):\n")
            for u, amt in agg.get("top_users_by_spend", {}).items():
                f.write(f"  - User {u}: ${amt}\n")
                
            f.write("\n--- UNIVARIATE ANALYSIS (NUMERIC) ---\n")
            for col, stats in eda.get("univariate_numeric", {}).items():
                f.write(f"{col.capitalize()}:\n")
                if "error" in stats:
                    f.write(f"  Error: {stats['error']}\n")
                else:
                    miss_cnt = missing_per_col.get(col) or missing_per_col.get(col.capitalize()) or missing_per_col.get(col.lower(), 0)
                    f.write(f"  Missing Values: {miss_cnt}\n")
                    f.write(f"  Min: {stats.get('min')}, Max: {stats.get('max')}, Mean: {stats.get('mean')}, Median: {stats.get('median')}\n")
                    f.write(f"  Std Dev: {stats.get('std_dev')}, 25%: {stats.get('25%')}, 75%: {stats.get('75%')}\n")
                    f.write(f"  Outliers (IQR method): {stats.get('outliers_count')}\n")
            
            f.write("\n--- UNIVARIATE ANALYSIS (CATEGORICAL) ---\n")
            for col, stats in eda.get("univariate_categorical", {}).items():
                f.write(f"{col.capitalize()}:\n")
                if "error" in stats:
                    f.write(f"  Error: {stats['error']}\n")
                else:
                    miss_cnt = missing_per_col.get(col) or missing_per_col.get(col.capitalize()) or missing_per_col.get(col.lower(), 0)
                    f.write(f"  Missing Values: {miss_cnt}\n")
                    for k, v in stats.get("counts", {}).items():
                        perc = stats.get("percents", {}).get(k, 0)
                        f.write(f"  - {k}: {v} ({perc}%)\n")

            f.write("\n--- BIVARIATE ANALYSIS ---\n")
            age_st = bi.get("age_stats", {})
            f.write("Age Group Stats:\n")
            for k, v in age_st.get('buckets', {}).items():
                f.write(f"  - {k}:\n")
                f.write(f"      Users: {v['user_count']} | Avg Logins: {v['avg_logins']} | Avg Amount: ${v['avg_amount']}\n")
                f.write(f"      Transactions: {v['success']} success, {v['failed']} failed, {v['unknown_status']} unknown\n")
                f.write(f"      Missing Values: {v['missing']}\n")
                
            f.write("\nDepartment Stats:\n")
            for d, s in agg.get("department_stats", {}).items():
                m_cnt = missing_per_dept.get(d, 0)
                f.write(f"  - {d}:\n")
                f.write(f"      Users: {s['user_count']} | Logins: {s['total_logins']} | Avg Amount: ${s['avg_amount']} | Avg Age: {s['avg_age']}\n")
                f.write(f"      Transactions: {s['success']} success, {s['failed']} failed, {s['unknown_status']} unknown\n")
                f.write(f"      Missing Values: {m_cnt}\n")
                
            f.write("\nTransaction Status vs Avg Login Count:\n")
            for k, v in bi.get("status_vs_avg_login", {}).items():
                f.write(f"  - {k}: {v} logins\n")
                
            f.write("\nCorrelations Matrix:\n")
            corr_matrix = bi.get('correlation_matrix', {})
            if "error" in corr_matrix:
                f.write(f"  Error computing correlation matrix: {corr_matrix['error']}\n")
            else:
                cols = ["Age", "login_count", "amount"]
                f.write(" " * 15 + "".join(f"{c:>15}" for c in cols) + "\n")
                for c1 in cols:
                    f.write(f"{c1:<15}")
                    for c2 in cols:
                        f.write(f"{corr_matrix.get(c1, {}).get(c2, 0):>15.4f}")
                    f.write("\n")

    @staticmethod
    def write_json(filename, data):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def write_quality_report(ml_data):
        with open("data_quality_report.txt", "w", encoding="utf-8") as f:
            f.write("=== DATA QUALITY & ANOMALY REPORT ===\n\n")
            anomalous = [u for u in ml_data if u['anomaly_flag'] == 1 or u['suspicious_flag'] == 1]
            f.write(f"Total Anomalous/Suspicious Users Detected: {len(anomalous)}\n\n")
            for u in anomalous:
                f.write(f"User ID: {u['user_id']}\nDepartment: {u['department']}\n")
                if u['suspicious_flag'] == 1:
                    f.write(f"Suspicious Reasons: {u['suspicious_reason']}\n")
                if u['anomaly_flag'] == 1:
                    f.write(f"Anomaly Reasons: {u['anomaly_reason']}\n")
                f.write(f"{'-'*40}\n")


class DataPipeline:
    """Facade: Orchestrates the entire data pipeline flow cleanly."""
    def run(self, emp_file, trans_file):
        print("Loading data...")
        emp = DataLoader.load(emp_file, schema=EmployeeSchema)
        trans = DataLoader.load(trans_file, schema=TransactionSchema)
        
        print("Cleaning and processing data...")
        raw_mapped, cleaned = DataCleaner(emp, trans).process()
        
        print("Performing Exploratory Data Analysis (EDA)...")
        eda_results = MetricsAggregator.calculate_eda(raw_mapped, cleaned, emp)
        
        print("Engineering features & preparing ML dataset...")
        ml_ready = FeatureEngineer(cleaned, emp).build_features()
        
        print("Generating reports...")
        ReportWriter.write_json("cleaned_data.json", cleaned)
        ReportWriter.write_analysis_summary(eda_results)
        ReportWriter.write_json("features_ready.json", ml_ready)
        ReportWriter.write_quality_report(ml_ready)
        print("Pipeline execution completed successfully.")

if __name__ == "__main__":
    DataPipeline().run("employees_realistic.csv", "transactions_realistic.json")
