# Competitive Programming Collection

A simple tool to organize DSA exercises.

## Structure

- `problems/` - Solutions organized by category
- `categories/` - Problems grouped by algorithm type
- `new_problem.py` - Script to add new problems
- `run_category_tests.py` - Script to run all solutions of a category

## Usage

```bash
python3 scripts/new_problem.py "<title>"
```
Example:
```bash
python3 scripts/new_problem.py "Two Sum"
```
## Test

```bash
python3 scripts/run_category_tests.py "<category>"
```
Example:
```bash
python3 scripts/run_category_tests.py linked-list
```

## Features

- Organizes by platform (LeetCode, Codeforces, etc.)
- Creates consistent IDs (LC0001, CF1234)
- Generates C++ templates
- Categorizes by algorithm type
