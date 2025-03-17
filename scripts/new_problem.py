#!/usr/bin/env python3
import os
import sys
import re

def create_problem_directory(problem_id, problem_title, difficulty, platform=None):
    # Get platform if not provided
    if platform is None:
        platforms = ["leetcode", "codeforces", "atcoder", "hackerrank", "spoj", "other"]
        print("\nSelect problem platform:")
        for i, plat in enumerate(platforms):
            print(f"{i+1}. {plat.capitalize()}")
        
        while True:
            try:
                choice = int(input("\nSelect platform (number): "))
                if 1 <= choice <= len(platforms):
                    platform = platforms[choice-1]
                    break
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a number.")
    
    # Format the ID based on platform
    if platform.lower() == "leetcode":
        # Format the problem ID with leading zeros for LeetCode
        formatted_id = f"{int(problem_id):04d}"
        composed_id = formatted_id
    elif platform.lower() == "codeforces":
        # For Codeforces, use format like CF1234A
        composed_id = f"CF{problem_id}"
    elif platform.lower() == "atcoder":
        # For AtCoder, use format like ABC123_A
        composed_id = problem_id  # Assume proper format like ABC123_A
    else:
        # For other platforms, use the ID as-is
        composed_id = problem_id
    
    # Create a slug from the title
    slug = re.sub(r'[^a-z0-9]', '-', problem_title.lower())
    slug = re.sub(r'-+', '-', slug).strip('-')
    
    # Create directory name with platform prefix
    dir_name = f"{composed_id}-{slug}"
    dir_path = os.path.join("problems", dir_name)
    
    # Create the directory if it doesn't exist
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory: {dir_path}")
    
    # Create README.md with template
    readme_path = os.path.join(dir_path, "README.md")
    with open(readme_path, "w") as f:
        f.write(f"# {composed_id}. {problem_title}\n\n")
        f.write(f"Platform: {platform.capitalize()}\n")
        f.write(f"Difficulty: {difficulty}\n\n")
        f.write("## Problem Description\n<!-- Copy the problem description here -->\n\n")
        f.write("## Approach\n<!-- Describe your approach to solving the problem -->\n\n")
        f.write("## Complexity Analysis\n- Time Complexity: O(?)\n- Space Complexity: O(?)\n\n")
        f.write("## Notes\n<!-- Any additional notes or insights -->\n")
    
    # Create empty C++ solution file
    solution_path = os.path.join(dir_path, "solution.cpp")
    with open(solution_path, "w") as f:
        f.write("""#include <bits/stdc++.h>

#define int long long int
#define F first
#define S second
#define pb push_back 

using namespace std;


int32_t main(){

#ifndef ONLINE_JUDGE
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
#endif

    return 0;
}""")
    
    # Create empty input and output files
    input_path = os.path.join(dir_path, "input.txt")
    output_path = os.path.join(dir_path, "output.txt")
    
    with open(input_path, "w") as f:
        f.write("// Test input goes here\n")
    
    with open(output_path, "w") as f:
        f.write("// Expected output goes here\n")
    
    print(f"Created problem template for {composed_id}. {problem_title}")
    
    # Prompt user to select a category
    print("\nAvailable Categories:")
    categories = [
        "arrays", "strings", "hash-table", "dynamic-programming", 
        "math", "greedy", "sorting", "binary-search", "tree", 
        "depth-first-search", "breadth-first-search", "graph",
        "backtracking", "stack", "queue", "heap", "linked-list", 
        "sliding-window", "two-pointers", "bit-manipulation", "design"
    ]
    
    # Create categories directory if it doesn't exist
    if not os.path.exists("categories"):
        os.makedirs("categories")
    
    # Display available categories
    for i, category in enumerate(categories):
        print(f"{i+1}. {category}")
    
    # Allow custom category option
    print(f"{len(categories)+1}. Add custom category")
    print(f"{len(categories)+2}. Skip adding to a category")
    
    # Get user selection
    while True:
        try:
            choice = int(input("\nSelect category (number): "))
            if 1 <= choice <= len(categories):
                category = categories[choice-1]
                break
            elif choice == len(categories)+1:
                category = input("Enter custom category name: ").strip().lower()
                # Format custom category name
                category = re.sub(r'[^a-z0-9]', '-', category)
                category = re.sub(r'-+', '-', category).strip('-')
                break
            elif choice == len(categories)+2:
                print("Skipping category assignment.")
                return
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a number.")
    
    # Update category file
    category_file = os.path.join("categories", f"{category}.md")
    
    # Create category file if it doesn't exist
    if not os.path.exists(category_file):
        with open(category_file, "w") as f:
            f.write(f"# {category.capitalize().replace('-', ' ')} Problems\n\n")
    
    # Append to category file
    with open(category_file, "a") as f:
        f.write(f"- [{composed_id}. {problem_title}](../problems/{dir_name}/README.md) - {platform.capitalize()}, {difficulty}\n")
    
    print(f"Added to {category} category")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python new_problem.py <problem_id> <problem_title> <difficulty> [platform]")
        sys.exit(1)
    
    problem_id = sys.argv[1]
    problem_title = sys.argv[2]
    difficulty = sys.argv[3] if len(sys.argv) > 3 else "Medium"
    platform = sys.argv[4] if len(sys.argv) > 4 else None
    
    create_problem_directory(problem_id, problem_title, difficulty, platform)

