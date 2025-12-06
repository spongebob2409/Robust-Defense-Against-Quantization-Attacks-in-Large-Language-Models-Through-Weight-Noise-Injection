# Git Commands to Push PyTorchTest2 to GitHub

## Prerequisites
1. Install Git: https://git-scm.com/download/win
2. Configure Git (first time only):
```powershell
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## Quick Push (Run these commands)

### Option 1: Using the provided script
```powershell
cd C:\codes\PyTorchTest2
.\git_push.bat
```

### Option 2: Manual commands
```powershell
# Navigate to the folder
cd C:\codes\PyTorchTest2

# Initialize git repository
git init

# Add remote repository
git remote add origin https://github.com/spongebob2409/Robust-Defense-Against-Quantization-Attacks-in-Large-Language-Models-Through-Weight-Noise-Injection.git

# Add all files (respecting .gitignore)
git add .

# Commit changes
git commit -m "Add quantization evaluation project with FP16, INT8, and AWQ 4-bit results"

# Push to GitHub (try main branch first)
git push -u origin main

# If main doesn't work, try master
git push -u origin master
```

## Authentication

If you're prompted for credentials, you have two options:

### Option 1: Personal Access Token (Recommended)
1. Go to: https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Select scopes: `repo` (full control)
4. Copy the token
5. Use token as password when prompted

### Option 2: GitHub CLI
```powershell
# Install GitHub CLI first
winget install GitHub.cli

# Authenticate
gh auth login
```

## Troubleshooting

### If repository already exists on GitHub:
```powershell
# Pull first to merge
git pull origin main --allow-unrelated-histories

# Then push
git push -u origin main
```

### If branch name is different:
```powershell
# Check current branch
git branch

# Rename to main if needed
git branch -M main

# Push
git push -u origin main
```

### Large files warning:
If you get warnings about large JSON files, you can either:
1. Keep them (they're important evaluation results)
2. Use Git LFS:
```powershell
git lfs install
git lfs track "*.json"
git add .gitattributes
git commit -m "Add Git LFS for large files"
```

## What Will Be Uploaded

Based on .gitignore, these files will be uploaded:
- ✅ All Python scripts (.py files)
- ✅ README.md and documentation
- ✅ Results JSON files (unless you modify .gitignore)
- ✅ EVALUATION_SUMMARY.md
- ❌ env1/ folder (virtual environment - excluded)
- ❌ __pycache__/ folders (excluded)
- ❌ .vscode/ settings (excluded)

## Repository URL
https://github.com/spongebob2409/Robust-Defense-Against-Quantization-Attacks-in-Large-Language-Models-Through-Weight-Noise-Injection

## Next Steps After Push
1. Visit your GitHub repository
2. Add repository description
3. Add topics/tags for discoverability
4. Update LICENSE if needed
5. Consider adding GitHub Actions for CI/CD
