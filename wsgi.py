"""
WSGI Entry Point for Gunicorn / Render
=======================================
Used by: gunicorn wsgi:app
"""

import os
from src.web_app import create_app

app = create_app()

if __name__ == "__main__":
    # Fallback for local testing: python wsgi.py
    from src.web_app import run_web
    run_web()
