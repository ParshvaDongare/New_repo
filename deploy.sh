#!/bin/bash

# Railway Deployment Script
# This script helps prepare and deploy the Policy Intelligence API to Railway

echo "🚀 Railway Deployment Script for Policy Intelligence API"
echo "======================================================"

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "❌ Error: Git repository not found. Please initialize git first:"
    echo "   git init"
    echo "   git add ."
    echo "   git commit -m 'Initial commit'"
    exit 1
fi

# Check if all required files exist
echo "📋 Checking required files..."

required_files=("app.py" "requirements.txt" "Procfile" "railway.json" "runtime.txt")
missing_files=()

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -ne 0 ]; then
    echo "❌ Missing required files:"
    for file in "${missing_files[@]}"; do
        echo "   - $file"
    done
    exit 1
fi

echo "✅ All required files present"

# Check if changes are committed
if [ -n "$(git status --porcelain)" ]; then
    echo "⚠️  Warning: You have uncommitted changes"
    echo "   Please commit your changes before deploying:"
    echo "   git add ."
    echo "   git commit -m 'Update for deployment'"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Push to GitHub if remote exists
if git remote get-url origin >/dev/null 2>&1; then
    echo "📤 Pushing to GitHub..."
    git push origin main
    echo "✅ Pushed to GitHub"
else
    echo "⚠️  No GitHub remote found. Please add your GitHub repository:"
    echo "   git remote add origin https://github.com/yourusername/policy-intelligence-api.git"
    echo "   git push -u origin main"
fi

echo ""
echo "🎯 Next Steps:"
echo "1. Go to https://railway.app"
echo "2. Click 'New Project'"
echo "3. Select 'Deploy from GitHub repo'"
echo "4. Connect your GitHub account"
echo "5. Select this repository"
echo "6. Add environment variables in Railway dashboard"
echo "7. Add PostgreSQL database"
echo ""
echo "📖 See DEPLOYMENT.md for detailed instructions"
echo "📋 See README.md for API documentation"
echo ""
echo "🔗 Useful links:"
echo "   - Railway Dashboard: https://railway.app"
echo "   - Railway Docs: https://docs.railway.app"
echo "   - Your Repository: https://github.com/yourusername/policy-intelligence-api" 