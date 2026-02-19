"""
VectorizerE entry point (legacy).
Delegates to ingestion.vectorizer_e.
Run: python vectorizerE.py <path_to_file.md>
"""

from ingestion.vectorizer_e import main

if __name__ == "__main__":
    main()
