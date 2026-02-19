"""
Email pipeline entry point.
Delegates to email.pipeline for all logic.
Run: python email_pipeline.py batch|poll|collection [options]
"""

from email_ingestion.pipeline import main

if __name__ == "__main__":
    main()
