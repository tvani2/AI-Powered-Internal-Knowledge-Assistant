from typing import Dict, Any, Optional

import pandas as pd

from tools.database_tool import DatabaseTool


class DatabaseHandler:
    """
    Database-only utilities extracted from IntelligentAgent for SQL execution and formatting.
    """

    def __init__(self) -> None:
        self.db_tool = DatabaseTool()

    def execute_database_query(self, sql: str) -> Dict[str, Any]:
        """Execute a database query and return results"""
        try:
            return self.db_tool.execute_sql(sql)
        except Exception as e:
            return {"error": str(e), "rows": [], "count": 0}

    def get_schema(self) -> Dict[str, Any]:
        """Expose database schema for SQL generation context."""
        return self.db_tool.get_schema()

    def try_execute_for_validation(self, sql: str) -> bool:
        """Attempt to execute SQL to validate it; return True if OK, False otherwise."""
        try:
            _ = self.db_tool.execute_sql(sql)
            return True
        except Exception:
            return False

    def format_database_response(self, results: dict, limit: int = None) -> str:
        if "error" in results:
            return f"<p style='color:red;'>Database error: {results['error']}</p>"
        
        rows = results.get("rows", [])
        if not rows:
            return "<p>No data found matching your query.</p>"
        
        df = pd.DataFrame(rows)

        # Apply client-requested limit if given
        if limit is not None and limit < len(df):
            df = df.head(limit)
        
        # Clean up column names for better display
        column_rename = {}
        for col in df.columns:
            if col == 'total_revenue':
                column_rename[col] = 'Total Revenue'
            elif col == 'employee_count':
                column_rename[col] = 'Employee Count'
            elif col == 'sales_count':
                column_rename[col] = 'Sales Count'
            elif col == 'hire_date':
                column_rename[col] = 'Hire Date'
            elif col == 'customer_name':
                column_rename[col] = 'Customer Name'
            elif col == 'budget':
                column_rename[col] = 'Budget'
            elif col == 'project_name':
                column_rename[col] = 'Project Name'
            elif col == 'manager_name':
                column_rename[col] = 'Manager Name'
            else:
                # Capitalize first letter of each word
                column_rename[col] = col.replace('_', ' ').title()
        
        df = df.rename(columns=column_rename)
        
        # Format numeric values nicely
        for col in df.columns:
            if ('Revenue' in col or 'Budget' in col) and df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "")
            elif df[col].dtype in ['float64', 'int64'] and col not in ['Revenue', 'Budget']:
                df[col] = df[col].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "")
        
        styled = (
            df.style
            .set_table_styles(
                [
                    {"selector": "table",
                    "props": "border-collapse: separate; border-spacing: 0; border: none; "
                            "border-radius: 12px; overflow: hidden; "
                            "font-family: 'Segoe UI', sans-serif; font-size: 14px; margin: 15px 0; "
                            "box-shadow: 0 8px 25px rgba(0,0,0,0.1), 0 4px 10px rgba(0,0,0,0.05);"},
                    {"selector": "thead th",
                    "props": "background: #007AFF; color: white; "
                            "padding: 14px 12px; text-align: left; font-weight: 600; border: none; "
                            "text-shadow: 0 1px 3px rgba(0,0,0,0.3); font-size: 13px; letter-spacing: 0.5px;"},
                    {"selector": "thead th:first-child",
                    "props": "border-top-left-radius: 15px;"},
                    {"selector": "thead th:last-child", 
                    "props": "border-top-right-radius: 15px;"},
                    {"selector": "tbody td",
                    "props": "padding: 12px; border: none; border-bottom: 1px solid #e9ecef; text-align: left; background-color: white; transition: all 0.2s ease;"},
                    {"selector": "tbody tr",
                    "props": "background-color: white; transition: all 0.2s ease;"},
                    {"selector": "tbody tr:hover",
                    "props": "background-color: #f8f9fa; transform: translateY(-1px); box-shadow: 0 2px 8px rgba(0,0,0,0.1);"},
                    {"selector": "tbody tr:last-child td:first-child",
                    "props": "border-bottom-left-radius: 15px;"},
                    {"selector": "tbody tr:last-child td:last-child",
                    "props": "border-bottom-right-radius: 15px;"},
                ]
            )
            .hide(axis="index")
        )
        
        return styled.to_html()
