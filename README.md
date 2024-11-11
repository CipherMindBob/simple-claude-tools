# Claude Calculator Tool

A beginner-friendly calculator that uses Claude AI to solve both simple math problems and complex word problems. Perfect for learning how to integrate AI into Python applications!

## What Does This Tool Do?
This calculator can:
- Solve basic math (like 5 + 3)
- Understand word problems ("If I have 10 apples and eat 3...")
- Handle money calculations ("What's the total cost of 5 items at $3 each?")
- Format answers nicely ($15.00, 7 apples, etc.)
- Retry automatically if something goes wrong

## Before You Start
You'll need:
- Python 3.8 or newer installed on your computer
  - Not sure? Open terminal/command prompt and type: `python --version`
  - If needed, download Python from [python.org](https://www.python.org/downloads/)
- A text editor (like VS Code, Sublime Text, or even Notepad)
- An Anthropic API key (we'll show you how to get this)
- Basic knowledge of using terminal/command prompt

## Step-by-Step Setup Guide

### 1. Get Your API Key
Before anything else, let's get your API key:
1. Go to [console.anthropic.com](https://console.anthropic.com/)
2. Sign up for an account (it's free!)
3. Click on "API Keys" in the dashboard
4. Create a new key and copy it (keep this safe and private!)

⚠️ IMPORTANT SECURITY STEPS:
- Never share your API key or commit it to GitHub
- Never include it directly in your code
- Always use a .env file to store your key
- Always use a .gitignore file to prevent accidentally sharing your key

To protect your API key:
1. Create a .gitignore file:
```bash
# For Mac/Linux:
touch .gitignore

# For Windows:
type nul > .gitignore
```

2. Add these lines to your .gitignore file:
```
.env
venv/
__pycache__/
*.pyc
.DS_Store
```

3. Create your .env file:
```bash
# For Mac/Linux:
touch .env

# For Windows:
type nul > .env
```

4. Add your API key to the .env file:
```
ANTHROPIC_API_KEY=your_api_key_here
```

5. Verify your security:
```bash
# This should NOT show .env in the output
git status
```

If you see .env in the git status output, check your .gitignore file!

### 2. Download the Code
Open your terminal/command prompt and type:
```bash
# Clone (download) the code
git clone https://github.com/CipherMindBob/simple-claude-tools.git

# Move into the project folder
cd simple-claude-tools
```

### 3. Set Up Your Work Environment
```bash
# Create a virtual environment (like a private workspace for this project)
# For Mac/Linux:
python3 -m venv venv
source venv/bin/activate

# For Windows:
python -m venv venv
.\venv\Scripts\activate

# Your prompt should change to show (venv) at the start
# This means you're in the virtual environment!
```

### 4. Install Required Tools
```bash
# Install everything the calculator needs
pip install -r requirements.txt
```

### 5. Set Up Your API Key
```bash
# Create a .env file
# For Mac/Linux:
touch .env

# For Windows:
type nul > .env
```

Now:
1. Open the .env file in your text editor
2. Add your API key like this (no spaces around =):
   ```
   ANTHROPIC_API_KEY=your_api_key_here
   ```
3. Save and close the file

### 6. Test It Out!
```bash
python calculator.py
```

You should see the calculator run some test problems. If you see results like "5 + 3 = 8", it's working!

## Try Your Own Calculations
After testing, you can use the calculator in your own Python code:
```python
from calculator import call_claude_with_calculator

# Try some calculations:
print(call_claude_with_calculator("What is 1234 times 5678?"))
print(call_claude_with_calculator("If I have 145 apples and give away 37, how many do I have left?"))
print(call_claude_with_calculator("A store sold 156 items at $13 each. What was the total?"))
```

## Common Problems and Solutions

### "ModuleNotFoundError: No module named 'anthropic'"
- Did you activate your virtual environment? Look for (venv) at the start of your prompt
- If not sure, run the activate command again
- Try installing requirements again: `pip install -r requirements.txt`

### "Can't find API key" or Similar Errors
- Check your .env file is in the same folder as calculator.py
- Make sure your API key is copied correctly
- No quotes around the API key
- No spaces around the = sign

### Virtual Environment Issues
- Make sure you're in the project folder
- On Windows, if you can't activate, run:
  ```bash
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```
- Try creating a new virtual environment if problems persist

### Still Stuck?
1. Check if Python is installed: `python --version`
2. Make sure you're in the right folder: `ls` or `dir` should show calculator.py
3. Try the steps again from the beginning
4. Create an issue on GitHub - we're here to help!

## Want to Learn More?
- Check out [Anthropic's Claude documentation](https://docs.anthropic.com/claude/docs)
- Learn about [Python virtual environments](https://docs.python.org/3/tutorial/venv.html)
- Explore [Python string formatting](https://docs.python.org/3/tutorial/inputoutput.html)

## License
MIT License - see LICENSE file for details. Feel free to use this code for learning and your own projects!