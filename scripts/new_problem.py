#!/usr/bin/env python3
import os
import sys
import re

def create_problem_directory(problem_title):
    categories = [
        "arrays", "strings", "hash-table", "dynamic-programming", 
        "math", "greedy", "sorting", "binary-search", "tree", 
        "depth-first-search", "breadth-first-search", "graph",
        "backtracking", "stack", "queue", "heap", "linked-list", "doubly-linked-list", 
        "sliding-window", "two-pointers", "bit-manipulation", "design"
    ]

    if not os.path.exists("categories"):
        os.makedirs("categories")

    print("\nAvailable Categories:")
    for i, category in enumerate(categories):
        print(f"{i+1}. {category}")
    print(f"{len(categories)+1}. Add custom category")
    print(f"{len(categories)+2}. Skip category")

    while True:
        try:
            choice = int(input("\nSelect category (number): "))
            if 1 <= choice <= len(categories):
                category = categories[choice-1]
                break
            elif choice == len(categories)+1:
                category = input("Enter custom category name: ").strip().lower()
                category = re.sub(r'[^a-z0-9]', '-', category)
                category = re.sub(r'-+', '-', category).strip('-')
                break
            elif choice == len(categories)+2:
                print("Skipping category assignment.")
                return
            else:
                print("Invalid choice. Try again.")
        except ValueError:
            print("Please enter a number.")

    # Create unique slug: <category>-<title>
    slug = f"{re.sub(r'[^a-z0-9]', '-', problem_title.lower())}"
    slug = re.sub(r'-+', '-', slug).strip('-')

    # Create directory: problems/<category>/<slug>
    dir_path = os.path.join("problems", category, slug)
    os.makedirs(dir_path, exist_ok=True)

    # README.md
    readme_path = os.path.join(dir_path, "README.md")
    with open(readme_path, "w") as f:
        f.write(f"# {problem_title}\n\n")
        f.write("## Problem Description\n<!-- Copy the problem description here -->\n\n")
        f.write("## Approach\n<!-- Describe your approach -->\n\n")
        f.write("## Complexity\n- Time: O(?)\n- Space: O(?)\n")

    # solution.py
    solution_path = os.path.join(dir_path, "solution.py")
    with open(solution_path, "w") as f:
        f.write("""class Solution:
    def solve(self, *args, **kwargs):
        # Implement your core logic here
        return

def test_solution():
    sol = Solution()
    
    # Example 1
    result1 = sol.solve()
    print("Result 1:", result1)
    # assert result1 == expected_value

    # Example 2
    result2 = sol.solve()
    print("Result 2:", result2)
    # assert result2 == expected_value

    print("All test cases passed!")

if __name__ == "__main__":
    test_solution()
""")

    print(f"Created problem directory for {problem_title} in category '{category}'")

    # Update master categories file
    categories_file = os.path.join("categories", "categories.md")
    if not os.path.exists(categories_file):
        with open(categories_file, "w") as f:
            f.write("# Problem Categories\n\n")

    # Read current content
    with open(categories_file, "r") as f:
        content = f.read()

    # Check if category section exists
    category_header = f"## {category.capitalize().replace('-', ' ')}"
    if category_header not in content:
        content += f"\n{category_header}\n"

    # Add problem entry
    problem_entry = f"- [{problem_title}](../problems/{category}/{slug}/README.md)\n"
    if problem_entry not in content:
        content += problem_entry

    # Write back
    with open(categories_file, "w") as f:
        f.write(content)

    print(f"Added to categories.md under '{category}'")


def create_problem_directory_heavy_weight(problem_id, problem_title, difficulty, platform=None):
    # Get platform if not provided
    if platform is None:
        platforms = ["leetcode", "crackingthecode", "udemy", "codeforces", "atcoder", "hackerrank", "spoj", "geeksforgeeks", "other"]
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
    
    # Format ID
    formatted_id = f"{int(problem_id):04d}"
    prefix_map = {
        "leetcode": "LC",
        "codeforces": "CF",
        "atcoder": "AC",
        "hackerrank": "HR",
        "udemy": "UD",
        "spoj": "SP",
        "geeksforgeeks": "GFG",
        "crackingthecode": "CtCI"
    }
    composed_id = f"{prefix_map.get(platform.lower(), 'OT')}{formatted_id}"


    # Language selection
    languages = ["cpp", "python"]
    print("\nSelect language:")
    for i, lang in enumerate(languages):
        print(f"{i+1}. {lang.upper()}")
    while True:
        try:
            lang_choice = int(input("\nSelect language (number): "))
            if 1 <= lang_choice <= len(languages):
                language = languages[lang_choice - 1]
                break
            else:
                print("Invalid choice. Try again.")
        except ValueError:
            print("Please enter a number.")

    
    # Slug from title
    slug = re.sub(r'[^a-z0-9]', '-', problem_title.lower())
    slug = re.sub(r'-+', '-', slug).strip('-')
    dir_name = f"{composed_id}-{slug}"
    
    # Create platform directory within problems if it doesn't exist
    platform_dir = os.path.join("problems", platform.lower())
    if not os.path.exists(platform_dir):
        os.makedirs(platform_dir)
        print(f"Created platform directory: {platform_dir}")
    
    # Platform directory
    platform_dir = os.path.join("problems", platform.lower())
    os.makedirs(platform_dir, exist_ok=True)
    
    # Problem directory
    dir_path = os.path.join(platform_dir, dir_name)
    os.makedirs(dir_path, exist_ok=True)
    
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
    
    
        # Generate solution file based on language
    if language == "cpp":
        solution_path = os.path.join(dir_path, "solution.cpp")
        with open(solution_path, "w") as f:
            f.write("""#include <bits/stdc++.h>
using namespace std;

#define FAST_IO ios::sync_with_stdio(false); cin.tie(nullptr); cout.tie(nullptr);
#define ll long long
#define vi vector<int>
#define vvi vector<vector<int>>
#define pii pair<int, int>
#define pb push_back
#define all(x) x.begin(), x.end()

#ifndef ONLINE_JUDGE
void setIO() {
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
}
#else
void setIO() {}
#endif

void solve() {
    int n, r;
    cin >> n >> r;
    vi nums(n);
    for (int& x : nums) cin >> x;

    cout << "---\\n";
}

int main() {
    FAST_IO;
    setIO();

    int T;
    cin >> T;
    while (T--) {
        solve();
    }

    return 0;
}
""")
    elif language == "python":
        solution_path = os.path.join(dir_path, "solution.py")
        with open(solution_path, "w") as f:
            f.write("""#!/usr/bin/env python3
import sys
import os

def read_input():
    if os.path.exists("input.txt"):
        sys.stdin = open("input.txt", "r")
    if os.path.exists("output.txt"):
        sys.stdout = open("output.txt", "w")

def read_non_comment_line():
    while True:
        line = sys.stdin.readline()
        if not line:
            return None  # End of input
        line = line.strip()
        if line and not line.startswith("//"):
            return line

class Solution:
    def solve_case(self, n, r, nums):
        # Problem-solving logic here
        # Return results, do NOT print
        result = f"Processed n={n}, r={r}, nums={nums}"
        return result

def main():
    read_input()
    t_line = read_non_comment_line()
    if t_line is None:
        print("No input found.")
        return
    try:
        t = int(t_line)
    except ValueError:
        print(f"Expected integer for number of test cases, got: {t_line}")
        return

    sol = Solution()
    for _ in range(t):
        line1 = read_non_comment_line()
        if line1 is None:
            print("Missing input for case")
            continue
        n, r = map(int, line1.split())

        line2 = read_non_comment_line()
        if line2 is None:
            print("Missing numbers line for case")
            continue
        nums = list(map(int, line2.split()))

        result = sol.solve_case(n, r, nums)
        print(result)

if __name__ == "__main__":
    main()
""")
    
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
    
    # Append to category file with updated relative path
    with open(category_file, "a") as f:
        # Update relative path to reflect new directory structure
        f.write(f"- [{composed_id}. {problem_title}](../problems/{platform.lower()}/{dir_name}/README.md) - {platform.capitalize()}, {difficulty}\n")
    
    print(f"Added to {category} category")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python new_problem.py <problem_title>")
        sys.exit(1)

    problem_title = sys.argv[1]
    create_problem_directory(problem_title)
