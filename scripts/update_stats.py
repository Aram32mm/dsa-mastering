#!/usr/bin/env python3
import os
import json
import re
from datetime import datetime
from collections import defaultdict, Counter

def update_stats():
    problems_dir = "problems"
    
    # Initialize stats dictionary
    stats = {
        "total": 0,
        "easy": 0,
        "medium": 0,
        "hard": 0,
        "platforms": defaultdict(int),
        "categories": defaultdict(int),
        "recent_problems": [],
        "monthly_progress": defaultdict(int),
        "last_updated": datetime.now().strftime("%Y-%m-%d")
    }
    
    # Get all problem directories
    all_problems = []
    if os.path.exists(problems_dir):
        for problem_dir in os.listdir(problems_dir):
            dir_path = os.path.join(problems_dir, problem_dir)
            if not os.path.isdir(dir_path):
                continue
            
            readme_path = os.path.join(dir_path, "README.md")
            if not os.path.exists(readme_path):
                continue
            
            # Read README content
            with open(readme_path, "r") as f:
                content = f.read()
            
            # Extract problem details
            problem_title_match = re.search(r'# (.*?)\n', content)
            problem_title = problem_title_match.group(1) if problem_title_match else problem_dir
            
            difficulty_match = re.search(r'Difficulty: (.*?)\n', content, re.IGNORECASE)
            difficulty = difficulty_match.group(1).lower() if difficulty_match else "unknown"
            
            platform_match = re.search(r'Platform: (.*?)\n', content, re.IGNORECASE)
            platform = platform_match.group(1).lower() if platform_match else "unknown"
            
            # Get last modified time for sorting recent problems
            solution_path = os.path.join(dir_path, "solution.cpp")
            last_modified = os.path.getmtime(solution_path) if os.path.exists(solution_path) else 0
            
            # Add to all problems list
            all_problems.append({
                "title": problem_title,
                "difficulty": difficulty,
                "platform": platform,
                "dir": problem_dir,
                "last_modified": last_modified
            })
            
            # Update counts
            stats["total"] += 1
            if "easy" in difficulty.lower():
                stats["easy"] += 1
            elif "medium" in difficulty.lower():
                stats["medium"] += 1
            elif "hard" in difficulty.lower():
                stats["hard"] += 1
            
            stats["platforms"][platform] += 1
            
            # Get month from last modified time
            modified_date = datetime.fromtimestamp(last_modified)
            month_key = modified_date.strftime("%Y-%m")
            stats["monthly_progress"][month_key] += 1
    
    # Sort problems by last modified time (most recent first)
    all_problems.sort(key=lambda x: x["last_modified"], reverse=True)
    
    # Get the 5 most recent problems
    stats["recent_problems"] = all_problems[:5]
    
    # Count problems by category
    categories_dir = "categories"
    if os.path.exists(categories_dir):
        for category_file in os.listdir(categories_dir):
            if not category_file.endswith(".md"):
                continue
            
            category_name = os.path.splitext(category_file)[0]
            category_path = os.path.join(categories_dir, category_file)
            
            with open(category_path, "r") as f:
                content = f.read()
                # Count links in the category file
                link_count = len(re.findall(r'\[.*?\]\(.*?\)', content))
                stats["categories"][category_name] = link_count
    
    # Save stats to JSON
    with open("stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)
    
    # Update README.md
    update_readme(stats)
    
    print(f"Updated stats: {stats['total']} total problems")
    return stats

def update_readme(stats):
    try:
        with open("README.md", "r") as f:
            readme = f.read()
    except FileNotFoundError:
        # If README doesn't exist, create it from scratch
        readme = """# Problem Solutions

My personal repository of Data Structures and Algorithms solutions from various platforms.

## 📊 Stats

## 📝 Recent Solutions

## 📂 Categories

## 📚 Platforms

## 🎯 Goals
- [ ] Solve a problem daily
- [ ] Complete 100 problems
- [ ] Master Dynamic Programming

## 🛠️ Scripts

## 📈 Progress Tracking

## 📅 Monthly Progress

## 💡 Study Notes

## 📝 Resources
"""
    
    # Update stats section
    stats_section = f"""## 📊 Stats
- **Total problems solved**: {stats['total']}
- **Easy**: {stats['easy']}
- **Medium**: {stats['medium']}
- **Hard**: {stats['hard']}
"""
    stats_pattern = r"## 📊 Stats\n.*?(?=\n##|\Z)"
    readme = re.sub(stats_pattern, stats_section.strip(), readme, flags=re.DOTALL)
    
    # Update recent solutions section
    recent_section = "## 📝 Recent Solutions\n"
    if stats["recent_problems"]:
        for problem in stats["recent_problems"]:
            title = problem["title"]
            difficulty = problem["difficulty"].capitalize()
            platform = problem["platform"].capitalize()
            dir_name = problem["dir"]
            recent_section += f"- [{title}](problems/{dir_name}/README.md) - {platform}, {difficulty}\n"
    else:
        recent_section += "No problems solved yet.\n"
    
    recent_pattern = r"## 📝 Recent Solutions\n.*?(?=\n##|\Z)"
    readme = re.sub(recent_pattern, recent_section.strip(), readme, flags=re.DOTALL)
    
    # Update progress tracking section
    progress_section = f"""## 📈 Progress Tracking
![Problem Count](https://img.shields.io/badge/Problems%20Solved-{stats['total']}-brightgreen)
![Easy Count](https://img.shields.io/badge/Easy-{stats['easy']}-success)
![Medium Count](https://img.shields.io/badge/Medium-{stats['medium']}-orange)
![Hard Count](https://img.shields.io/badge/Hard-{stats['hard']}-red)
"""
    progress_pattern = r"## 📈 Progress Tracking\n.*?(?=\n##|\Z)"
    readme = re.sub(progress_pattern, progress_section.strip(), readme, flags=re.DOTALL)
    
    # Update monthly progress section
    monthly_section = "## 📅 Monthly Progress\n"
    if stats["monthly_progress"]:
        # Sort months chronologically (newest first)
        sorted_months = sorted(stats["monthly_progress"].items(), reverse=True)
        for month, count in sorted_months:
            # Format month as "July 2023" instead of "2023-07"
            try:
                date_obj = datetime.strptime(month, "%Y-%m")
                formatted_month = date_obj.strftime("%B %Y")
            except:
                formatted_month = month
                
            monthly_section += f"- **{formatted_month}**: {count} problems\n"
    else:
        monthly_section += "No monthly data available yet.\n"
    
    monthly_pattern = r"## 📅 Monthly Progress\n.*?(?=\n##|\Z)"
    readme = re.sub(monthly_pattern, monthly_section.strip(), readme, flags=re.DOTALL)
    
    # Write updated README
    with open("README.md", "w") as f:
        f.write(readme)

if __name__ == "__main__":
    update_stats()
