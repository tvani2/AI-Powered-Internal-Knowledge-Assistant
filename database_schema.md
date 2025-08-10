# Enhanced Database Schema Documentation

## Overview
The AI-Powered Internal Knowledge Assistant uses a SQLite database with three main tables to store comprehensive company information about employees, sales and projects. The schema includes foreign key relationships, additional columns for richer queries and strategic indexes for performance.

## Tables

### 1. Employees Table
Stores information about company employees, their roles and hierarchical relationships.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | INTEGER | PRIMARY KEY, AUTOINCREMENT | Unique employee identifier |
| name | TEXT | NOT NULL | Employee's full name |
| department | TEXT | NOT NULL | Department the employee works in |
| role | TEXT | NOT NULL | Job title/role within department |
| email | TEXT | NOT NULL, UNIQUE | Employee's email address |
| hire_date | DATE | NOT NULL | Date employee was hired |
| salary | DECIMAL(10,2) | NOT NULL | Employee's annual salary |
| manager_id | INTEGER | FOREIGN KEY | ID of the employee's manager (self-referencing) |
| phone | TEXT | | Employee's phone number |
| address | TEXT | | Employee's address |

**Departments:** Engineering, Sales, Marketing, HR, Finance, Operations, Product, Support, Legal, Research

**Sample Roles by Department:**
- Engineering: Software Engineer, Senior Engineer, Engineering Manager, DevOps Engineer, QA Engineer, Data Engineer, Frontend Developer, Backend Developer
- Sales: Sales Representative, Sales Manager, Account Executive, Sales Director, Business Development, Sales Operations
- Marketing: Marketing Specialist, Marketing Manager, Content Creator, Brand Manager, Digital Marketing, Product Marketing
- HR: HR Specialist, HR Manager, Recruiter, Benefits Coordinator, Talent Acquisition, Employee Relations
- Finance: Financial Analyst, Accountant, Finance Manager, Controller, Treasury Analyst, Auditor
- Operations: Operations Manager, Process Analyst, Operations Specialist, Supply Chain Manager, Logistics Coordinator
- Product: Product Manager, Product Owner, Product Analyst, UX Designer, UI Designer, Product Marketing
- Support: Support Specialist, Support Manager, Technical Support, Customer Success, Implementation Specialist
- Legal: Legal Counsel, Compliance Officer, Contract Manager, IP Specialist
- Research: Research Scientist, Data Scientist, Research Analyst, Lab Manager

**Salary Ranges by Department:**
- Engineering: $70,000 - $180,000
- Sales: $50,000 - $150,000
- Marketing: $45,000 - $120,000
- HR: $40,000 - $100,000
- Finance: $50,000 - $130,000
- Operations: $45,000 - $110,000
- Product: $60,000 - $140,000
- Support: $35,000 - $90,000
- Legal: $70,000 - $160,000
- Research: $60,000 - $150,000

### 2. Sales Table
Tracks product sales with employee attribution and customer information.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | INTEGER | PRIMARY KEY, AUTOINCREMENT | Unique sale identifier |
| date | DATE | NOT NULL | Date of the sale |
| product | TEXT | NOT NULL | Name of the product sold |
| category | TEXT | NOT NULL | Product category |
| revenue | DECIMAL(10,2) | NOT NULL | Revenue generated from the sale |
| employee_id | INTEGER | NOT NULL, FOREIGN KEY | ID of the employee who made the sale |
| customer_name | TEXT | NOT NULL | Name of the customer |
| customer_email | TEXT | | Customer's email address |
| quantity | INTEGER | NOT NULL, DEFAULT 1 | Quantity of products sold |

**Product Categories:** Electronics, Computers, Accessories, Gaming, Audio, Storage, Smart Home, Wearables

**Sample Products:**
- Computers: Laptop Pro X1, Monitor 4K, Graphics Card RTX
- Electronics: Smartphone Galaxy S, Tablet Air, Drone Pro
- Gaming: Gaming Console, Gaming Mouse, VR Headset, Mechanical Keyboard Pro
- Audio: Wireless Headphones, Bluetooth Speaker, Microphone Pro, Wireless Earbuds
- Accessories: USB-C Cable, Wireless Mouse, Webcam HD
- Storage: SSD 1TB, RAM 16GB
- Smart Home: Smart Speaker, Smart Home Hub
- Wearables: Smart Watch, Fitness Tracker

**Revenue Ranges by Category:**
- Computers: $800 - $2,500
- Electronics: $300 - $1,200
- Gaming: $400 - $600
- Audio: $50 - $300
- Accessories: $20 - $150
- Storage: $80 - $200
- Smart Home: $100 - $400
- Wearables: $150 - $500

### 3. Projects Table
Manages company projects with comprehensive details and tracking.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | INTEGER | PRIMARY KEY, AUTOINCREMENT | Unique project identifier |
| name | TEXT | NOT NULL | Project name |
| description | TEXT | | Project description |
| manager_id | INTEGER | NOT NULL, FOREIGN KEY | ID of the employee managing the project |
| status | TEXT | NOT NULL | Current project status |
| start_date | DATE | NOT NULL | Project start date |
| deadline | DATE | | Project deadline |
| budget | DECIMAL(12,2) | | Project budget |
| priority | TEXT | DEFAULT 'Medium' | Project priority level |

**Project Statuses:** Active, Completed, On Hold, Planning, Delayed, Cancelled

**Priority Levels:** Low, Medium, High, Critical

**Sample Projects:** Project Atlas, Project Phoenix, Digital Transformation, Cloud Migration, AI Implementation, Security Upgrade, Mobile App Development, Website Redesign, Data Analytics Platform, CRM Integration

**Budget Ranges by Project Type:**
- Digital Transformation: $50,000 - $500,000
- Migration/Implementation: $30,000 - $300,000
- Development/Launch: $20,000 - $200,000
- Other Projects: $10,000 - $100,000

## Relationships

### Foreign Key Constraints
- **Projects → Employees**: `manager_id` references `employees.id`
- **Sales → Employees**: `employee_id` references `employees.id`
- **Employees → Employees**: `manager_id` references `employees.id` (self-referencing for hierarchy)

### Relationship Rules
- Each project has exactly one manager (employee)
- Each sale is attributed to exactly one employee
- Each employee can manage multiple projects
- Each employee can make multiple sales
- Employees can have managers (hierarchical structure)
- Each employee can manage multiple other employees

## Indexes

### Performance Optimization
The database includes strategic indexes on frequently queried columns:

- **Employees:** `department`, `role`
- **Sales:** `category`, `date`, `employee_id`
- **Projects:** `status`, `manager_id`, `deadline`

These indexes significantly improve query performance for common operations like:
- Finding employees by department
- Filtering sales by category or date
- Looking up projects by status or manager
- Date-based queries and reporting

## Data Population

The database is populated with realistic sample data using the Faker library:
- **100 employees** across 10 departments with various roles and hierarchical relationships
- **300 sales records** spanning the last year with employee attribution and customer details
- **50 projects** with different statuses, priorities and realistic budgets

## Usage Examples

### Sample Queries

**Find project managers with their project counts:**
```sql
SELECT e.name, e.department, COUNT(p.id) as project_count, AVG(p.budget) as avg_budget
FROM employees e
JOIN projects p ON e.id = p.manager_id
GROUP BY e.id, e.name, e.department
ORDER BY project_count DESC;
```

**Sales performance by employee and department:**
```sql
SELECT e.name, e.department, SUM(s.revenue) as total_revenue, COUNT(s.id) as sales_count
FROM employees e
JOIN sales s ON e.id = s.employee_id
GROUP BY e.id, e.name, e.department
ORDER BY total_revenue DESC;
```

**Projects by status and priority:**
```sql
SELECT status, priority, COUNT(*) as count, AVG(budget) as avg_budget
FROM projects
GROUP BY status, priority
ORDER BY status, priority;
```

**Employee hierarchy (who reports to whom):**
```sql
SELECT e.name, e.role, m.name as manager_name, m.role as manager_role
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.id
ORDER BY e.department, e.manager_id;
```

**Sales summary by category and quarter:**
```sql
SELECT 
    category,
    strftime('%Y-Q%m', date) as quarter,
    COUNT(*) as sales_count,
    SUM(revenue) as total_revenue,
    AVG(revenue) as avg_revenue
FROM sales
GROUP BY category, quarter
ORDER BY quarter, total_revenue DESC;
```

## Database File
- **File:** `company.db`
- **Type:** SQLite 3 with foreign key constraints enabled
- **Location:** Project root directory
- **Size:** Approximately 2-3 MB after population
- **Features:** Foreign keys, indexes, realistic data relationships

## Setup
Run the setup script to create and populate the database:
```bash
python setup_database.py
```

This will:
1. Create the database file `company.db` if it doesn't exist
2. Create all tables with proper schema and foreign key constraints
3. Create performance indexes on frequently queried columns
4. Populate tables with realistic sample data (100+ employees, 300+ sales, 50+ projects)
5. Display sample data and demonstrate sample queries
6. Show database summary statistics

## FastAPI Integration
The database is designed to be immediately pluggable into FastAPI applications with:
- Proper foreign key relationships for data integrity
- Indexed columns for fast query performance
- Rich data structure for complex business queries
- Realistic sample data for testing and development
