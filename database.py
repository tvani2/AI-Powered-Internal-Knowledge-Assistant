#!/usr/bin/env python3
"""
Database setup and population for AI-Powered Internal Knowledge Assistant
"""

import sqlite3
import pandas as pd
from faker import Faker
import os
from datetime import datetime, timedelta
import random

# Initialize Faker
fake = Faker()

class DatabaseManager:
    def __init__(self, db_path="company.db"):
        self.db_path = db_path
        self.conn = None
        
    def connect(self):
        """Create connection to SQLite database"""
        self.conn = sqlite3.connect(self.db_path)
        # Enable foreign key constraints
        self.conn.execute("PRAGMA foreign_keys = ON")
        return self.conn
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def create_tables(self):
        """Create the database tables with enhanced schema"""
        cursor = self.conn.cursor()
        
        # Employees table with additional columns
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS employees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                department TEXT NOT NULL,
                role TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                hire_date DATE NOT NULL,
                salary DECIMAL(10,2) NOT NULL,
                manager_id INTEGER,
                phone TEXT,
                address TEXT,
                FOREIGN KEY (manager_id) REFERENCES employees (id)
            )
        ''')
        
        # Sales table with employee and customer information
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sales (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                product TEXT NOT NULL,
                category TEXT NOT NULL,
                revenue DECIMAL(10,2) NOT NULL,
                employee_id INTEGER NOT NULL,
                customer_name TEXT NOT NULL,
                customer_email TEXT,
                quantity INTEGER NOT NULL DEFAULT 1,
                FOREIGN KEY (employee_id) REFERENCES employees (id)
            )
        ''')
        
        # Projects table with additional project details
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                manager_id INTEGER NOT NULL,
                status TEXT NOT NULL,
                start_date DATE NOT NULL,
                deadline DATE,
                budget DECIMAL(12,2),
                priority TEXT DEFAULT 'Medium',
                FOREIGN KEY (manager_id) REFERENCES employees (id)
            )
        ''')
        
        # Query logs table for debugging
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS query_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_query TEXT NOT NULL,
                query_type TEXT,
                confidence REAL,
                sql_generated TEXT,
                documents_searched TEXT,
                response TEXT,
                processing_time REAL,
                error_message TEXT,
                user_id TEXT,
                session_id TEXT
            )
        ''')
        
        # Create indexes for frequently searched columns
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_employees_department ON employees(department)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_employees_role ON employees(role)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sales_category ON sales(category)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sales_date ON sales(date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sales_employee ON sales(employee_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_projects_status ON projects(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_projects_manager ON projects(manager_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_projects_deadline ON projects(deadline)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_query_logs_timestamp ON query_logs (timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_query_logs_user ON query_logs (user_id)')
        
        self.conn.commit()
        print("Database tables and indexes created successfully!")
    
    def populate_employees(self, count=100):
        """Populate employees table with realistic data"""
        departments = ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance', 'Operations', 'Product', 'Support', 'Legal', 'Research']
        roles = {
            'Engineering': ['Software Engineer', 'Senior Engineer', 'Engineering Manager', 'DevOps Engineer', 'QA Engineer', 'Data Engineer', 'Frontend Developer', 'Backend Developer'],
            'Sales': ['Sales Representative', 'Sales Manager', 'Account Executive', 'Sales Director', 'Business Development', 'Sales Operations'],
            'Marketing': ['Marketing Specialist', 'Marketing Manager', 'Content Creator', 'Brand Manager', 'Digital Marketing', 'Product Marketing'],
            'HR': ['HR Specialist', 'HR Manager', 'Recruiter', 'Benefits Coordinator', 'Talent Acquisition', 'Employee Relations'],
            'Finance': ['Financial Analyst', 'Accountant', 'Finance Manager', 'Controller', 'Treasury Analyst', 'Auditor'],
            'Operations': ['Operations Manager', 'Process Analyst', 'Operations Specialist', 'Supply Chain Manager', 'Logistics Coordinator'],
            'Product': ['Product Manager', 'Product Owner', 'Product Analyst', 'UX Designer', 'UI Designer', 'Product Marketing'],
            'Support': ['Support Specialist', 'Support Manager', 'Technical Support', 'Customer Success', 'Implementation Specialist'],
            'Legal': ['Legal Counsel', 'Compliance Officer', 'Contract Manager', 'IP Specialist'],
            'Research': ['Research Scientist', 'Data Scientist', 'Research Analyst', 'Lab Manager']
        }
        
        # Salary ranges by department
        salary_ranges = {
            'Engineering': (70000, 180000),
            'Sales': (50000, 150000),
            'Marketing': (45000, 120000),
            'HR': (40000, 100000),
            'Finance': (50000, 130000),
            'Operations': (45000, 110000),
            'Product': (60000, 140000),
            'Support': (35000, 90000),
            'Legal': (70000, 160000),
            'Research': (60000, 150000)
        }
        
        employees_data = []
        manager_ids = []  # Track potential managers
        
        for i in range(count):
            department = random.choice(departments)
            role = random.choice(roles[department])
            name = fake.name()
            email = fake.email()
            
            # Generate realistic hire date (within last 10 years)
            hire_date = fake.date_between(start_date='-10y', end_date='today')
            
            # Generate realistic salary based on department and role
            base_salary_range = salary_ranges[department]
            if 'Manager' in role or 'Director' in role or 'Senior' in role:
                salary = random.uniform(base_salary_range[1] * 0.8, base_salary_range[1])
            else:
                salary = random.uniform(base_salary_range[0], base_salary_range[1] * 0.8)
            
            # Some employees have managers (hierarchical structure)
            manager_id = None
            if i > 20 and random.random() < 0.7:  # 70% of employees after first 20 have managers
                manager_id = random.choice(manager_ids) if manager_ids else None
            
            phone = fake.phone_number()
            address = fake.address()
            
            employees_data.append((name, department, role, email, hire_date, round(salary, 2), manager_id, phone, address))
            
            # Track potential managers (employees with certain roles or higher salaries)
            if 'Manager' in role or 'Director' in role or salary > base_salary_range[1] * 0.7:
                manager_ids.append(i + 1)  # +1 because SQLite auto-increment starts at 1
        
        cursor = self.conn.cursor()
        cursor.executemany('''
            INSERT INTO employees (name, department, role, email, hire_date, salary, manager_id, phone, address)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', employees_data)
        
        self.conn.commit()
        print(f"Populated {count} employees!")
    
    def populate_sales(self, count=300):
        """Populate sales table with realistic data"""
        products = [
            'Laptop Pro X1', 'Smartphone Galaxy S', 'Wireless Headphones', 'Tablet Air',
            'Gaming Console', 'Smart Watch', 'Bluetooth Speaker', 'USB-C Cable',
            'Wireless Mouse', 'Mechanical Keyboard', 'Monitor 4K', 'Webcam HD',
            'Microphone Pro', 'Graphics Card RTX', 'SSD 1TB', 'RAM 16GB',
            'Gaming Mouse', 'Mechanical Keyboard Pro', 'Wireless Earbuds', 'Smart Speaker',
            'Fitness Tracker', 'VR Headset', 'Drone Pro', 'Smart Home Hub'
        ]
        
        categories = ['Electronics', 'Computers', 'Accessories', 'Gaming', 'Audio', 'Storage', 'Smart Home', 'Wearables']
        product_categories = {
            'Laptop Pro X1': 'Computers',
            'Smartphone Galaxy S': 'Electronics',
            'Wireless Headphones': 'Audio',
            'Tablet Air': 'Electronics',
            'Gaming Console': 'Gaming',
            'Smart Watch': 'Wearables',
            'Bluetooth Speaker': 'Audio',
            'USB-C Cable': 'Accessories',
            'Wireless Mouse': 'Accessories',
            'Mechanical Keyboard': 'Accessories',
            'Monitor 4K': 'Computers',
            'Webcam HD': 'Accessories',
            'Microphone Pro': 'Audio',
            'Graphics Card RTX': 'Computers',
            'SSD 1TB': 'Storage',
            'RAM 16GB': 'Storage',
            'Gaming Mouse': 'Gaming',
            'Mechanical Keyboard Pro': 'Gaming',
            'Wireless Earbuds': 'Audio',
            'Smart Speaker': 'Smart Home',
            'Fitness Tracker': 'Wearables',
            'VR Headset': 'Gaming',
            'Drone Pro': 'Electronics',
            'Smart Home Hub': 'Smart Home'
        }
        
        # Get employee IDs for sales representatives
        cursor = self.conn.cursor()
        cursor.execute('SELECT id FROM employees WHERE department = "Sales" OR role LIKE "%Sales%"')
        sales_employee_ids = [row[0] for row in cursor.fetchall()]
        
        if not sales_employee_ids:
            # Fallback to any employee if no sales employees found
            cursor.execute('SELECT id FROM employees LIMIT 20')
            sales_employee_ids = [row[0] for row in cursor.fetchall()]
        
        sales_data = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        for i in range(count):
            date = fake.date_between(start_date=start_date, end_date=end_date)
            product = random.choice(products)
            category = product_categories[product]
            employee_id = random.choice(sales_employee_ids)
            customer_name = fake.name()
            customer_email = fake.email()
            quantity = random.randint(1, 5)
            
            # Generate realistic revenue based on product category and quantity
            if category == 'Computers':
                base_price = random.uniform(800, 2500)
            elif category == 'Electronics':
                base_price = random.uniform(300, 1200)
            elif category == 'Gaming':
                base_price = random.uniform(400, 600)
            elif category == 'Audio':
                base_price = random.uniform(50, 300)
            elif category == 'Accessories':
                base_price = random.uniform(20, 150)
            elif category == 'Storage':
                base_price = random.uniform(80, 200)
            elif category == 'Smart Home':
                base_price = random.uniform(100, 400)
            else:  # Wearables
                base_price = random.uniform(150, 500)
            
            revenue = base_price * quantity
            
            sales_data.append((date, product, category, round(revenue, 2), employee_id, customer_name, customer_email, quantity))
        
        cursor.executemany('''
            INSERT INTO sales (date, product, category, revenue, employee_id, customer_name, customer_email, quantity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', sales_data)
        
        self.conn.commit()
        print(f"Populated {count} sales records!")
    
    def populate_projects(self, count=50):
        """Populate projects table with realistic data"""
        project_names = [
            'Project Atlas', 'Project Phoenix', 'Project Mercury', 'Project Orion',
            'Project Nova', 'Project Eclipse', 'Project Aurora', 'Project Zenith',
            'Project Horizon', 'Project Stellar', 'Project Quantum', 'Project Fusion',
            'Project Genesis', 'Project Evolution', 'Project Innovation', 'Project Future',
            'Project Legacy', 'Project Vision', 'Project Destiny', 'Project Infinity',
            'Digital Transformation', 'Cloud Migration', 'AI Implementation', 'Security Upgrade',
            'Mobile App Development', 'Website Redesign', 'Data Analytics Platform', 'CRM Integration',
            'Supply Chain Optimization', 'Customer Experience Enhancement', 'Product Launch', 'Market Expansion',
            'Compliance Framework', 'Performance Optimization', 'Scalability Improvement', 'Disaster Recovery',
            'Training Program Development', 'Process Automation', 'Quality Assurance System', 'Reporting Dashboard',
            'API Development', 'Microservices Architecture', 'Database Optimization', 'Network Infrastructure',
            'Cybersecurity Enhancement', 'Business Intelligence', 'Workflow Management', 'Collaboration Platform'
        ]
        
        project_descriptions = [
            'Strategic initiative to modernize core business processes',
            'Implementation of cutting-edge technology solutions',
            'Customer-focused improvement project',
            'Operational efficiency enhancement',
            'Compliance and regulatory framework development',
            'Digital transformation and modernization',
            'Product development and market launch',
            'Infrastructure and technology upgrade',
            'Process optimization and automation',
            'Data analytics and business intelligence'
        ]
        
        statuses = ['Active', 'Completed', 'On Hold', 'Planning', 'Delayed', 'Cancelled']
        priorities = ['Low', 'Medium', 'High', 'Critical']
        
        # Get employee IDs who can be managers
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT id FROM employees 
            WHERE role LIKE "%Manager%" OR role LIKE "%Director%" OR role LIKE "%Senior%"
            LIMIT 30
        ''')
        manager_ids = [row[0] for row in cursor.fetchall()]
        
        if not manager_ids:
            # Fallback to any employee if no managers found
            cursor.execute('SELECT id FROM employees LIMIT 20')
            manager_ids = [row[0] for row in cursor.fetchall()]
        
        projects_data = []
        current_date = datetime.now()
        
        for i in range(count):
            name = project_names[i] if i < len(project_names) else f"Project {chr(65 + (i % 26))}{i}"
            description = random.choice(project_descriptions)
            manager_id = random.choice(manager_ids)
            status = random.choice(statuses)
            priority = random.choice(priorities)
            
            # Generate realistic dates
            start_date = fake.date_between(start_date='-2y', end_date='-6m')
            
            # Deadline depends on status
            if status == 'Completed':
                deadline = fake.date_between(start_date=start_date, end_date='-1m')
            elif status == 'Active':
                deadline = fake.date_between(start_date='+1m', end_date='+1y')
            elif status == 'Planning':
                deadline = fake.date_between(start_date='+3m', end_date='+2y')
            else:  # On Hold, Delayed, Cancelled
                deadline = fake.date_between(start_date='+6m', end_date='+3y')
            
            # Generate realistic budget based on project scope
            if 'Digital' in name or 'Transformation' in name:
                budget = random.uniform(50000, 500000)
            elif 'Migration' in name or 'Implementation' in name:
                budget = random.uniform(30000, 300000)
            elif 'Development' in name or 'Launch' in name:
                budget = random.uniform(20000, 200000)
            else:
                budget = random.uniform(10000, 100000)
            
            projects_data.append((name, description, manager_id, status, start_date, deadline, round(budget, 2), priority))
        
        cursor.executemany('''
            INSERT INTO projects (name, description, manager_id, status, start_date, deadline, budget, priority)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', projects_data)
        
        self.conn.commit()
        print(f"Populated {count} projects!")
    
    def populate_all_data(self):
        """Populate all tables with sample data"""
        print("Starting database population...")
        self.populate_employees(100)
        self.populate_sales(300)
        self.populate_projects(50)
        print("Database population completed!")
    
    def show_sample_data(self):
        """Display sample data from all tables"""
        print("\n=== SAMPLE DATA ===")
        
        # Employees
        print("\n--- Employees (first 5) ---")
        df_employees = pd.read_sql_query("SELECT * FROM employees LIMIT 5", self.conn)
        print(df_employees.to_string(index=False))
        
        # Sales
        print("\n--- Sales (first 5) ---")
        df_sales = pd.read_sql_query("SELECT * FROM sales LIMIT 5", self.conn)
        print(df_sales.to_string(index=False))
        
        # Projects
        print("\n--- Projects (first 5) ---")
        df_projects = pd.read_sql_query("SELECT * FROM projects LIMIT 5", self.conn)
        print(df_projects.to_string(index=False))
        
        # Summary statistics
        print("\n--- DATABASE SUMMARY ---")
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM employees")
        emp_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM sales")
        sales_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM projects")
        projects_count = cursor.fetchone()[0]
        
        print(f"Total Employees: {emp_count}")
        print(f"Total Sales Records: {sales_count}")
        print(f"Total Projects: {projects_count}")
        
        # Show some sample queries
        print("\n--- SAMPLE QUERIES ---")
        
        # Top performing sales employees
        print("\nTop 3 Sales Employees by Revenue:")
        df_top_sales = pd.read_sql_query('''
            SELECT e.name, e.department, SUM(s.revenue) as total_revenue, COUNT(s.id) as sales_count
            FROM employees e
            JOIN sales s ON e.id = s.employee_id
            GROUP BY e.id, e.name, e.department
            ORDER BY total_revenue DESC
            LIMIT 3
        ''', self.conn)
        print(df_top_sales.to_string(index=False))
        
        # Projects by status
        print("\nProjects by Status:")
        df_proj_status = pd.read_sql_query('''
            SELECT status, COUNT(*) as count, AVG(budget) as avg_budget
            FROM projects
            GROUP BY status
            ORDER BY count DESC
        ''', self.conn)
        print(df_proj_status.to_string(index=False))

    def log_query(self, user_query, query_type=None, confidence=None, sql_generated=None, 
                  documents_searched=None, response=None, processing_time=None, 
                  error_message=None, user_id=None, session_id=None):
        """Log a query for debugging purposes"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO query_logs 
            (user_query, query_type, confidence, sql_generated, documents_searched, 
             response, processing_time, error_message, user_id, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_query, query_type, confidence, sql_generated, 
            documents_searched, response, processing_time, error_message, 
            user_id, session_id
        ))
        
        conn.commit()
        conn.close()
        
    def get_query_logs(self, limit=50, user_id=None):
        """Retrieve query logs for debugging"""
        conn = sqlite3.connect(self.db_path)
        
        if user_id:
            query = '''
                SELECT * FROM query_logs 
                WHERE user_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            '''
            df = pd.read_sql_query(query, conn, params=(user_id, limit))
        else:
            query = '''
                SELECT * FROM query_logs 
                ORDER BY timestamp DESC 
                LIMIT ?
            '''
            df = pd.read_sql_query(query, conn, params=(limit,))
        
        conn.close()
        return df

def main():
    """Main function to set up and populate the database"""
    db = DatabaseManager()
    
    try:
        db.connect()
        db.create_tables()
        db.populate_all_data()
        db.show_sample_data()
    finally:
        db.close()

if __name__ == "__main__":
    main()
