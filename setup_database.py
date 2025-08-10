#!/usr/bin/env python3
"""
Database setup script for AI-Powered Internal Knowledge Assistant
This script creates the database schema and populates it with sample data.
"""

from database import DatabaseManager

def main():
    print("Setting up AI-Powered Internal Knowledge Assistant Database...")
    print("=" * 60)
    
    # Initialize database manager
    db = DatabaseManager()
    
    try:
        # Connect to database
        print("Connecting to database...")
        db.connect()
        
        # Create tables
        print("Creating database tables...")
        db.create_tables()
        
        # Populate with sample data
        print("Populating database with sample data...")
        db.populate_all_data()
        
        # Show sample data
        db.show_sample_data()
        
        print("\n" + "=" * 60)
        print("Database setup completed successfully!")
        print("You can now run the AI assistant application.")
        
    except Exception as e:
        print(f"Error during database setup: {e}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    main()
