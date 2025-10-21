import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import settings

print("=== Checking Config Settings ===")
print(f"PROJECT_NAME: {settings.PROJECT_NAME}")

# Check MySQL settings
mysql_attrs = ['MYSQL_USER', 'MYSQL_PASSWORD', 'MYSQL_HOST', 'MYSQL_PORT', 'MYSQL_DATABASE']
for attr in mysql_attrs:
    if hasattr(settings, attr):
        value = getattr(settings, attr)
        # Hide password for security
        display_value = '***' if 'PASSWORD' in attr else value
        print(f"✅ {attr}: {display_value}")
    else:
        print(f"❌ {attr}: MISSING")

print(f"DATABASE_URL: {settings.DATABASE_URL}")