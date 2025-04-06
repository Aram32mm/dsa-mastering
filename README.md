# Competitive Programming Collection

A simple tool to organize competitive programming solutions.

## Structure

- `problems/` - Solutions organized by platform
- `categories/` - Problems grouped by algorithm type
- `new_problem.py` - Script to add new problems

## Usage

```bash
python new_problem.py <id> "<title>" "<difficulty>" "<platform>"
```

Example:
```bash
python new_problem.py 1 "Two Sum" "Easy" "leetcode"
```

## Features

- Organizes by platform (LeetCode, Codeforces, etc.)
- Creates consistent IDs (LC0001, CF1234)
- Generates C++ templates
- Creates test files
- Categorizes by algorithm type
