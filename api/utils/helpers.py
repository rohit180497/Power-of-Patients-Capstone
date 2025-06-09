import random
import string
import uuid
from typing import Optional

def generate_verification_code(length: int = 6) -> str:
    """Generate a random verification code"""
    return ''.join(random.choices(string.digits, k=length))

def generate_uuid() -> str:
    """Generate a random UUID"""
    return str(uuid.uuid4())

def format_date_iso(date_obj) -> str:
    """Format a date object as ISO 8601 string"""
    if hasattr(date_obj, 'isoformat'):
        return date_obj.isoformat()
    return str(date_obj)

def row_to_dict(row) -> dict:
    """Convert a database row to a dictionary"""
    if hasattr(row, 'keys') and callable(row.keys):
        return {key: value for key, value in zip(row.keys(), row)}
    return dict(row)