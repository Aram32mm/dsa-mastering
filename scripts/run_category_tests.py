import os
import subprocess
import sys

def run_all_tests_in_category(category):
    base_dir = os.path.join("problems", category)
    if not os.path.exists(base_dir):
        print(f"Category folder '{base_dir}' does not exist.")
        return

    problem_dirs = [
        os.path.join(base_dir, d) for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    if not problem_dirs:
        print(f"No problems found in category '{category}'.")
        return

    print(f"Running tests for category: {category}")
    failed = []

    for problem_dir in problem_dirs:
        solution_file = os.path.join(problem_dir, "solution.py")
        if not os.path.exists(solution_file):
            print(f"Skipping (no solution.py): {problem_dir}")
            continue

        print(f"\nRunning: {solution_file}")
        try:
            result = subprocess.run(
                [sys.executable, solution_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
                text=True
            )

            if "All test cases passed!" in result.stdout:
                print("✅ Passed")
            else:
                print("❌ Failed or missing success message")
                print("Output:\n", result.stdout)
                failed.append(problem_dir)

            if result.stderr:
                print("⚠️ Errors:\n", result.stderr)

        except subprocess.TimeoutExpired:
            print(f"❌ Timeout running {solution_file}")
            failed.append(problem_dir)

    print("\n--- Summary ---")
    if failed:
        print(f"{len(failed)} failed:")
        for f in failed:
            print(f" - {f}")
    else:
        print("✅ All problems passed!")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_category_tests.py <category>")
    else:
        category = sys.argv[1]
        run_all_tests_in_category(category)
