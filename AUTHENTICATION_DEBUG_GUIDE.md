# 🔐 AUTHENTICATION DEBUG GUIDE: "Invalid token" Error Fix

## 🚨 Problem Identified

The "Invalid token" error with 401 Unauthorized indicates an authentication issue when trying to upload datasets. This can be caused by several factors.

## 🔍 Debugging Steps Applied

I've added comprehensive debugging to help identify the exact issue:

### 1. **Enhanced Authentication Debugging**
- ✅ Added debug logging to `get_current_user` function
- ✅ Added debug logging to `decode_access_token` function
- ✅ Created test endpoint `/api/test-auth` to verify authentication

### 2. **Debug Information Added**
The server will now print:
- 🔍 Token received (first 20 characters)
- 🔍 Token decode success/failure
- 🔍 User ID extraction
- 🔍 User lookup results
- ❌ Specific error messages for failures

## 🧪 Testing Steps

### Step 1: Test Authentication Endpoint
First, test if authentication is working at all:

```bash
# Test with a valid token (replace YOUR_TOKEN with actual token)
curl -H "Authorization: Bearer YOUR_TOKEN" http://localhost:8000/api/test-auth
```

### Step 2: Check Server Logs
When you try to upload a dataset, watch the server console for debug messages:

```
🔍 Debug: Received token: eyJhbGciOiJIUzI1NiIs...
🔍 Debug: Decoding token with secret key: your-super...
🔍 Debug: Token payload: {'sub': '1', 'iat': 1696..., 'exp': 1696...}
🔍 Debug: Extracted user_id: 1
✅ Debug: User found: your_username
```

## 🛠️ Common Issues and Solutions

### Issue 1: Token Not Being Sent
**Symptoms:** "🔍 Debug: No token received"
**Solution:** Check frontend authentication context

### Issue 2: Token Expired
**Symptoms:** "❌ Debug: JWT decode error: Signature has expired"
**Solution:** User needs to log in again

### Issue 3: Invalid Token Format
**Symptoms:** "❌ Debug: JWT decode error: Invalid token"
**Solution:** Token might be corrupted or malformed

### Issue 4: User Not Found
**Symptoms:** "❌ Debug: User not found for user_id: X"
**Solution:** Database issue or user was deleted

## 🔧 Immediate Fixes Applied

### 1. **Enhanced Error Handling**
- More specific error messages
- Better debugging information
- Graceful fallbacks

### 2. **Test Endpoint Added**
- `/api/test-auth` - Test authentication without file upload
- Returns user information if authentication works

## 🧪 How to Test the Fix

### Option 1: Use the Test Endpoint
1. **Login to your application** to get a fresh token
2. **Open browser developer tools** (F12)
3. **Go to Console tab**
4. **Run this command** (replace with your actual token):
   ```javascript
   fetch('http://localhost:8000/api/test-auth', {
     headers: {
       'Authorization': 'Bearer YOUR_TOKEN_HERE'
     }
   }).then(r => r.json()).then(console.log)
   ```

### Option 2: Check Server Logs
1. **Try uploading a file** in your application
2. **Watch the server console** for debug messages
3. **Look for the specific error** in the debug output

## 🎯 Expected Debug Output

### Successful Authentication:
```
🔍 Debug: Received token: eyJhbGciOiJIUzI1NiIs...
🔍 Debug: Decoding token with secret key: your-super...
🔍 Debug: Token payload: {'sub': '1', 'iat': 1696..., 'exp': 1696...}
🔍 Debug: Extracted user_id: 1
✅ Debug: User found: your_username
```

### Failed Authentication:
```
🔍 Debug: Received token: invalid_token...
🔍 Debug: Decoding token with secret key: your-super...
❌ Debug: JWT decode error: Invalid token
❌ Debug: Token decode failed - returning 401
```

## 🚨 Next Steps

1. **Try uploading a file** and check the server logs
2. **Look for the debug messages** to identify the specific issue
3. **Use the test endpoint** to isolate the authentication problem
4. **Share the debug output** so I can provide a targeted fix

## 🔄 Quick Fixes to Try

### If Token is Expired:
1. **Log out** from your application
2. **Log in again** to get a fresh token
3. **Try uploading** again

### If Token is Not Being Sent:
1. **Check browser developer tools** Network tab
2. **Look for Authorization header** in the upload request
3. **Verify the token** is stored in localStorage

### If User Not Found:
1. **Check if the user exists** in the database
2. **Try creating a new account** and test with that

The debug information will help us identify the exact cause and provide a targeted solution! 🔍
