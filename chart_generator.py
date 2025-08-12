#!/usr/bin/env python3
"""
Chart Generator for AI-Powered Internal Knowledge Assistant
Generates matplotlib charts for numeric data visualization
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
import base64
from typing import Dict, List, Optional, Tuple
import re

# Set style for better-looking charts
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ChartGenerator:
    """Generates charts for database query results"""
    
    def __init__(self):
        self.chart_types = {
            'bar': self._create_bar_chart,
            'line': self._create_line_chart,
            'pie': self._create_pie_chart,
            'scatter': self._create_scatter_chart
        }
    
    def generate_chart(self, data: str, query: str) -> Optional[str]:
        """Generate appropriate chart based on data and query"""
        try:
            # Parse the data string to extract numeric information
            df = self._parse_data_string(data)
            if df is None or df.empty:
                return None
            
            # Determine chart type based on query and data
            chart_type = self._determine_chart_type(query, df)
            
            # Generate the chart
            chart_func = self.chart_types.get(chart_type, self._create_bar_chart)
            return chart_func(df, query)
            
        except Exception as e:
            print(f"Error generating chart: {e}")
            return None
    
    def _parse_data_string(self, data: str) -> Optional[pd.DataFrame]:
        """Parse data string into DataFrame"""
        try:
            # Look for table-like data with | separators
            if '|' in data:
                lines = [line.strip() for line in data.split('\n') if line.strip() and '|' in line]
                if len(lines) < 2:
                    return None
                
                # Parse header and data
                header = [col.strip() for col in lines[0].split('|') if col.strip()]
                rows = []
                for line in lines[1:]:
                    cols = [col.strip() for col in line.split('|') if col.strip()]
                    if len(cols) == len(header):
                        rows.append(cols)
                
                if rows:
                    df = pd.DataFrame(rows, columns=header)
                    # Convert numeric columns
                    for col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='ignore')
                    return df
            
            return None
        except Exception:
            return None
    
    def _determine_chart_type(self, query: str, df: pd.DataFrame) -> str:
        """Determine the best chart type based on query and data"""
        query_lower = query.lower()
        
        # Check for time-based queries
        if any(word in query_lower for word in ['trend', 'over time', 'monthly', 'yearly']):
            return 'line'
        
        # Check for comparison queries
        if any(word in query_lower for word in ['top', 'best', 'highest', 'lowest', 'compare']):
            return 'bar'
        
        # Check for distribution queries
        if any(word in query_lower for word in ['distribution', 'percentage', 'proportion']):
            return 'pie'
        
        # Check for correlation queries
        if any(word in query_lower for word in ['correlation', 'relationship', 'vs']):
            return 'scatter'
        
        # Default to bar chart for most cases
        return 'bar'
    
    def _create_bar_chart(self, df: pd.DataFrame, query: str) -> str:
        """Create a bar chart"""
        try:
            plt.figure(figsize=(10, 6))
            
            # Find numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) == 0:
                return None
            
            # Use first numeric column for values, first non-numeric for categories
            value_col = numeric_cols[0]
            category_col = df.select_dtypes(exclude=['number']).columns[0] if len(df.select_dtypes(exclude=['number']).columns) > 0 else df.index
            
            # Sort by value for better visualization
            df_sorted = df.sort_values(value_col, ascending=False).head(10)
            
            plt.bar(range(len(df_sorted)), df_sorted[value_col])
            plt.xticks(range(len(df_sorted)), df_sorted[category_col], rotation=45, ha='right')
            plt.ylabel(value_col)
            plt.title(f"{query[:50]}...")
            plt.tight_layout()
            
            return self._save_chart_to_base64()
            
        except Exception as e:
            print(f"Error creating bar chart: {e}")
            return None
    
    def _create_line_chart(self, df: pd.DataFrame, query: str) -> str:
        """Create a line chart"""
        try:
            plt.figure(figsize=(10, 6))
            
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) == 0:
                return None
            
            value_col = numeric_cols[0]
            x_col = df.columns[0] if df.columns[0] != value_col else df.columns[1]
            
            plt.plot(df[x_col], df[value_col], marker='o')
            plt.xlabel(x_col)
            plt.ylabel(value_col)
            plt.title(f"{query[:50]}...")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            return self._save_chart_to_base64()
            
        except Exception as e:
            print(f"Error creating line chart: {e}")
            return None
    
    def _create_pie_chart(self, df: pd.DataFrame, query: str) -> str:
        """Create a pie chart"""
        try:
            plt.figure(figsize=(8, 8))
            
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) == 0:
                return None
            
            value_col = numeric_cols[0]
            category_col = df.select_dtypes(exclude=['number']).columns[0] if len(df.select_dtypes(exclude=['number']).columns) > 0 else df.index
            
            # Take top 8 categories for readability
            df_sorted = df.sort_values(value_col, ascending=False).head(8)
            
            plt.pie(df_sorted[value_col], labels=df_sorted[category_col], autopct='%1.1f%%')
            plt.title(f"{query[:50]}...")
            plt.tight_layout()
            
            return self._save_chart_to_base64()
            
        except Exception as e:
            print(f"Error creating pie chart: {e}")
            return None
    
    def _create_scatter_chart(self, df: pd.DataFrame, query: str) -> str:
        """Create a scatter chart"""
        try:
            plt.figure(figsize=(10, 6))
            
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) < 2:
                return None
            
            x_col, y_col = numeric_cols[0], numeric_cols[1]
            
            plt.scatter(df[x_col], df[y_col], alpha=0.6)
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title(f"{query[:50]}...")
            plt.tight_layout()
            
            return self._save_chart_to_base64()
            
        except Exception as e:
            print(f"Error creating scatter chart: {e}")
            return None
    
    def _save_chart_to_base64(self) -> str:
        """Save chart to base64 string"""
        try:
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            return f"data:image/png;base64,{image_base64}"
        except Exception as e:
            print(f"Error saving chart: {e}")
            plt.close()
            return None

# Global chart generator instance
chart_generator = ChartGenerator()

