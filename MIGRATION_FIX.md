# Database Migration Fix for "no such column: users.full_name" Error

## Problem
The application was throwing a `sqlite3.OperationalError: no such column: users.full_name` error during signup because the database schema was out of sync with the User model definition.

## Solution
This fix provides a safe migration that can be applied without stopping the server.

## Files Modified

### 1. `migrate_add_full_name.py` (NEW)
- Database migration script that safely adds the missing `full_name` column
- Works with both SQLite and MySQL
- Includes verification and rollback capabilities
- Can be run multiple times safely (idempotent)

### 2. `routers/auth.py` (MODIFIED)
- Updated signup function to explicitly set `full_name=None` when creating users
- Prevents future column reference errors

### 3. `schemas.py` (MODIFIED)
- Added `full_name: Optional[str] = None` to `UserOut` schema
- Ensures the field is properly returned in API responses

### 4. `run_migration.py` (NEW)
- Simple wrapper script to run the migration from the backend directory

## How to Apply the Fix

### Option 1: Run Migration Script (Recommended)
```bash
cd backend
python run_migration.py
```

### Option 2: Run Migration Directly
```bash
cd backend/app
python migrate_add_full_name.py
```

### Option 3: Manual SQL (If scripts don't work)
For SQLite:
```sql
ALTER TABLE users ADD COLUMN full_name VARCHAR(255);
```

For MySQL:
```sql
ALTER TABLE users ADD COLUMN full_name VARCHAR(255);
```

## Verification
After running the migration, the script will:
1. ✅ Verify the column was added successfully
2. ✅ Check that all expected columns are present in the users table
3. ✅ Display the current table structure

## Expected Output
```
🚀 Starting database migration for full_name column...
==================================================
🔍 Checking if full_name column exists in users table...
📝 Adding full_name column to users table...
✅ Successfully added full_name column to users table!
✅ Verification: full_name column confirmed in users table

==================================================
✅ Migration completed successfully!

🔍 Verifying users table structure...
Expected columns: ['id', 'email', 'username', 'hashed_password', 'full_name', 'created_at']
Existing columns: ['id', 'email', 'username', 'hashed_password', 'full_name', 'created_at']
✅ All expected columns are present!

🎉 You can now restart your application or the migration will be applied automatically.
```

## Safety Features
- **Idempotent**: Can be run multiple times safely
- **Transactional**: Uses database transactions with rollback on failure
- **Cross-platform**: Works with both SQLite and MySQL
- **Verification**: Automatically verifies the migration was successful
- **Non-destructive**: Only adds columns, doesn't modify existing data

## After Migration
1. The signup endpoint will work without errors
2. New users will have `full_name` set to `NULL` by default
3. Existing users will have `full_name` as `NULL`
4. The API will properly return the `full_name` field in responses

## Troubleshooting
If you encounter issues:
1. Check database connection settings in `config.py`
2. Ensure you have proper database permissions
3. Verify the database file exists (for SQLite)
4. Check that MySQL service is running (for MySQL)
