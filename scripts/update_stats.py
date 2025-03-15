#!/usr/bin/env python3
import os
import json
import re

def update_stats():
    problems_dir = "problems"
    stats = {"total": 0, "easy": 0, "medium": 0, "hard": 0}
    
    # Count problems by difficulty
    for problem_dir in os.listdir(problems_dir):
        if not os.path.isdir(os.path.join(problems_dir, problem_dir)):
            continue
        
        readme_path = os.path.join(problems_dir, problem_dir, "README.md")
        if not os.path.exists(readme_path):
            continue
        
        stats["total"] += 1
        
        # Extract difficulty from README
        with open(readme_path, "r") as f:
            content = f.read().lower()
            if "difficulty: easy" in content:
                stats["easy"] += 1
            elif "difficulty: medium" in content:
                stats["medium"] += 1
            elif "difficulty: hard" in content:
                stats["hard"] += 1
    
    # Save stats to JSON
    with open("stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    # Update README.md
    with open("README.md", "r") as f:
        readme = f.read()
    
    # Replace stats section
    stats_pattern = r"## Stats\n-.*?\n-.*?\n-.*?\n-.*?\n"
    stats_replacement = f"## Stats\n- Total problems solved: {stats['total']}\n- Easy: {stats['easy']}\n- Medium: {stats['medium']}\n- Hard: {stats['hard']}\n"
    
    updated_readme = re.sub(stats_pattern, stats_replacement, readme, flags=re.DOTALL)
    
    with open("README.md", "w") as f:
        f.write(updated_readme)
    
    print(f"Updated stats: {stats}")

if __name__ == "__main__":
    update_stats()
