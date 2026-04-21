import csv
import json
import random

# Generate realistic names to replace "Employee_1", etc.
# since the user wants names without digits to be valid.
first_names = ["John", "Jane", "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Heidi", "Ivan", "Judy", "Mallory", "Victor", "Peggy", "Trent"]
last_names = ["Smith", "Doe", "Johnson", "Brown", "Williams", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez"]

def get_random_name():
    return f"{random.choice(first_names)} {random.choice(last_names)}"

def corrupt_csv(input_path, output_path):
    print(f"Modifying {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Modify header
    new_rows = []
    fieldnames = ['user_id', 'department', 'name', 'last_login', 'Age', 'login_count']

    for i, row in enumerate(rows):
        new_row = {}
        new_row['user_id'] = row.get('emp_id', '')
        new_row['department'] = row.get('department', '')
        
        # Replace Employee_1 with a real name to pass the new strict validation
        # But corrupt ~10% of names with digits/symbols
        if random.random() < 0.10:
            new_row['name'] = get_random_name() + str(random.randint(1, 99)) + "!"
        else:
            new_row['name'] = get_random_name()
            
        new_row['last_login'] = row.get('date_of_joining', '')

        # Generate Age (valid 22-60) with ~15% corruption
        r_age = random.random()
        if r_age < 0.05:
            new_row['Age'] = "-5" # Negative
        elif r_age < 0.10:
            new_row['Age'] = "twenty-five" # String
        elif r_age < 0.15:
            new_row['Age'] = "" # Missing
        else:
            new_row['Age'] = str(random.randint(22, 60))

        # Generate login_count (valid 1-1000) with ~15% corruption
        r_login = random.random()
        if r_login < 0.05:
            new_row['login_count'] = "-10" # Negative
        elif r_login < 0.10:
            new_row['login_count'] = "few" # String
        elif r_login < 0.15:
            new_row['login_count'] = "" # Missing
        else:
            new_row['login_count'] = str(random.randint(1, 1000))

        new_rows.append(new_row)

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_rows)
    print("CSV modification complete.")

def modify_json(input_path, output_path):
    print(f"Modifying {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for record in data:
        if 'emp_id' in record:
            record['user_id'] = record.pop('emp_id')

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print("JSON modification complete.")

if __name__ == "__main__":
    corrupt_csv("employees_realistic.csv", "employees_realistic.csv")
    modify_json("transactions_realistic.json", "transactions_realistic.json")
