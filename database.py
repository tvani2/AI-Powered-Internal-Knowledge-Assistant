#!/usr/bin/env python3
"""
Database management module for AI-Powered Internal Knowledge Assistant.
Creates the SQLite database, tables, indexes, and populates with sample data.
"""

import os
import random
import sqlite3
from datetime import datetime, timedelta
from typing import List, Tuple

from faker import Faker


class DatabaseManager:
    """Manage SQLite database creation, schema, and sample data population."""

    def __init__(self, db_path: str = "company.db") -> None:
        self.db_path = db_path
        self.conn: sqlite3.Connection | None = None
        self.fake = Faker()

    def connect(self) -> None:
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    # ---------------- Schema -----------------
    def create_tables(self) -> None:
        assert self.conn is not None, "Database not connected"
        cur = self.conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS employees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                department TEXT NOT NULL,
                role TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                hire_date DATE NOT NULL,
                salary DECIMAL(10,2) NOT NULL,
                manager_id INTEGER,
                phone TEXT,
                address TEXT,
                FOREIGN KEY (manager_id) REFERENCES employees(id)
            )
            """
        )

        cur.execute(
            """
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
                FOREIGN KEY (employee_id) REFERENCES employees(id)
            )
            """
        )

        cur.execute(
            """
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
                FOREIGN KEY (manager_id) REFERENCES employees(id)
            )
            """
        )

        # Indexes
        cur.execute("CREATE INDEX IF NOT EXISTS idx_employees_department ON employees(department)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_employees_role ON employees(role)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_sales_category ON sales(category)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_sales_date ON sales(date)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_sales_employee ON sales(employee_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_projects_status ON projects(status)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_projects_manager ON projects(manager_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_projects_deadline ON projects(deadline)")

        self.conn.commit()

    # ------------- Sample Data ---------------
    def _random_department_and_role(self) -> Tuple[str, str]:
        departments = {
            "Engineering": [
                "Software Engineer",
                "Senior Engineer",
                "Engineering Manager",
                "DevOps Engineer",
                "QA Engineer",
                "Data Engineer",
                "Frontend Developer",
                "Backend Developer",
            ],
            "Sales": [
                "Sales Representative",
                "Sales Manager",
                "Account Executive",
                "Sales Director",
                "Business Development",
                "Sales Operations",
            ],
            "Marketing": [
                "Marketing Specialist",
                "Marketing Manager",
                "Content Creator",
                "Brand Manager",
                "Digital Marketing",
                "Product Marketing",
            ],
            "HR": [
                "HR Specialist",
                "HR Manager",
                "Recruiter",
                "Benefits Coordinator",
                "Talent Acquisition",
                "Employee Relations",
            ],
            "Finance": [
                "Financial Analyst",
                "Accountant",
                "Finance Manager",
                "Controller",
                "Treasury Analyst",
                "Auditor",
            ],
            "Operations": [
                "Operations Manager",
                "Process Analyst",
                "Operations Specialist",
                "Supply Chain Manager",
                "Logistics Coordinator",
            ],
            "Product": [
                "Product Manager",
                "Product Owner",
                "Product Analyst",
                "UX Designer",
                "UI Designer",
            ],
            "Support": [
                "Support Specialist",
                "Support Manager",
                "Technical Support",
                "Customer Success",
                "Implementation Specialist",
            ],
            "Legal": [
                "Legal Counsel",
                "Compliance Officer",
                "Contract Manager",
                "IP Specialist",
            ],
            "Research": [
                "Research Scientist",
                "Data Scientist",
                "Research Analyst",
                "Lab Manager",
            ],
        }
        department = random.choice(list(departments.keys()))
        role = random.choice(departments[department])
        return department, role

    def populate_employees(self, count: int = 100) -> List[int]:
        assert self.conn is not None
        cur = self.conn.cursor()

        employee_ids: List[int] = []
        # First pass: insert employees without managers
        for _ in range(count):
            department, role = self._random_department_and_role()
            hire_date = datetime.now() - timedelta(days=random.randint(30, 365 * 10))
            salary = round(random.uniform(40000, 180000), 2)
            name = self.fake.name()
            email = self.fake.unique.email()
            phone = self.fake.phone_number()
            address = self.fake.address().replace("\n", ", ")

            cur.execute(
                """
                INSERT INTO employees (name, department, role, email, hire_date, salary, phone, address)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    name,
                    department,
                    role,
                    email,
                    hire_date.date().isoformat(),
                    salary,
                    phone,
                    address,
                ),
            )
            employee_ids.append(cur.lastrowid)

        # Second pass: assign managers (10-30% employees are managers)
        manager_candidates = random.sample(employee_ids, k=max(1, int(len(employee_ids) * 0.2)))
        for emp_id in employee_ids:
            if random.random() < 0.7:  # 70% have a manager
                manager_id = random.choice(manager_candidates)
                if manager_id != emp_id:
                    cur.execute("UPDATE employees SET manager_id = ? WHERE id = ?", (manager_id, emp_id))

        self.conn.commit()
        return employee_ids

    def populate_sales(self, employee_ids: List[int], count: int = 300) -> None:
        assert self.conn is not None
        cur = self.conn.cursor()

        categories = {
            "Computers": ["Laptop Pro X1", "Monitor 4K", "Graphics Card RTX"],
            "Electronics": ["Smartphone Galaxy S", "Tablet Air", "Drone Pro"],
            "Gaming": ["Gaming Console", "Gaming Mouse", "VR Headset", "Mechanical Keyboard Pro"],
            "Audio": ["Wireless Headphones", "Bluetooth Speaker", "Microphone Pro", "Wireless Earbuds"],
            "Accessories": ["USB-C Cable", "Wireless Mouse", "Webcam HD"],
            "Storage": ["SSD 1TB", "RAM 16GB"],
            "Smart Home": ["Smart Speaker", "Smart Home Hub"],
            "Wearables": ["Smart Watch", "Fitness Tracker"],
        }

        for _ in range(count):
            category = random.choice(list(categories.keys()))
            product = random.choice(categories[category])
            date = datetime.now() - timedelta(days=random.randint(0, 365))
            revenue_ranges = {
                "Computers": (800, 2500),
                "Electronics": (300, 1200),
                "Gaming": (400, 600),
                "Audio": (50, 300),
                "Accessories": (20, 150),
                "Storage": (80, 200),
                "Smart Home": (100, 400),
                "Wearables": (150, 500),
            }
            low, high = revenue_ranges[category]
            revenue = round(random.uniform(low, high), 2)
            employee_id = random.choice(employee_ids)
            customer_name = self.fake.name()
            customer_email = self.fake.safe_email()
            quantity = random.randint(1, 5)

            cur.execute(
                """
                INSERT INTO sales (date, product, category, revenue, employee_id, customer_name, customer_email, quantity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    date.date().isoformat(),
                    product,
                    category,
                    revenue,
                    employee_id,
                    customer_name,
                    customer_email,
                    quantity,
                ),
            )

        self.conn.commit()

    def populate_projects(self, employee_ids: List[int], count: int = 50) -> None:
        assert self.conn is not None
        cur = self.conn.cursor()

        project_names = [
            "Project Atlas",
            "Project Phoenix",
            "Digital Transformation",
            "Cloud Migration",
            "AI Implementation",
            "Security Upgrade",
            "Mobile App Development",
            "Website Redesign",
            "Data Analytics Platform",
            "CRM Integration",
        ]
        statuses = ["Active", "Completed", "On Hold", "Planning", "Delayed", "Cancelled"]
        priorities = ["Low", "Medium", "High", "Critical"]

        for _ in range(count):
            name = random.choice(project_names)
            description = self.fake.paragraph(nb_sentences=3)
            manager_id = random.choice(employee_ids)
            status = random.choice(statuses)
            start_date = datetime.now() - timedelta(days=random.randint(0, 365))
            deadline = start_date + timedelta(days=random.randint(30, 240))
            budget = round(random.uniform(10000, 500000), 2)
            priority = random.choice(priorities)

            cur.execute(
                """
                INSERT INTO projects (name, description, manager_id, status, start_date, deadline, budget, priority)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    name,
                    description,
                    manager_id,
                    status,
                    start_date.date().isoformat(),
                    deadline.date().isoformat(),
                    budget,
                    priority,
                ),
            )

        self.conn.commit()

    # ------------- Orchestration -------------
    def populate_all_data(self) -> None:
        employee_ids = self.populate_employees()
        self.populate_sales(employee_ids)
        self.populate_projects(employee_ids)

    def show_sample_data(self) -> None:
        assert self.conn is not None
        cur = self.conn.cursor()
        print("Employees sample:")
        for row in cur.execute("SELECT id, name, department, role FROM employees LIMIT 5"):
            print(dict(row))
        print("Sales sample:")
        for row in cur.execute("SELECT id, date, product, revenue FROM sales LIMIT 5"):
            print(dict(row))
        print("Projects sample:")
        for row in cur.execute("SELECT id, name, status, budget FROM projects LIMIT 5"):
            print(dict(row))


def main() -> None:
    db = DatabaseManager()
    try:
        db.connect()
        db.create_tables()
        # Only populate if empty
        cur = db.conn.cursor()  # type: ignore[union-attr]
        cur.execute("SELECT COUNT(*) FROM employees")
        if cur.fetchone()[0] == 0:
            print("Populating database with sample data...")
            db.populate_all_data()
        db.show_sample_data()
        print("Database is ready at company.db")
    finally:
        db.close()


if __name__ == "__main__":
    main()
