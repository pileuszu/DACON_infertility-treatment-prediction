import os
import argparse
import subprocess

def setup_github(repo_name, username=None):
    """
    Set up a GitHub repository and push the code
    
    Args:
        repo_name: Name of the GitHub repository
        username: GitHub username (optional)
    """
    # Initialize git repository
    print(f"Initializing git repository...")
    subprocess.run(["git", "init"], check=True)
    
    # Add all files
    print(f"Adding files to git...")
    subprocess.run(["git", "add", "."], check=True)
    
    # Commit changes
    print(f"Committing changes...")
    subprocess.run(["git", "commit", "-m", "Initial commit"], check=True)
    
    # Create .gitignore if it doesn't exist
    if not os.path.exists(".gitignore"):
        print(f"Creating .gitignore file...")
        with open(".gitignore", "w") as f:
            f.write("# Python\n")
            f.write("__pycache__/\n")
            f.write("*.py[cod]\n")
            f.write("*$py.class\n")
            f.write("*.so\n")
            f.write(".Python\n")
            f.write("build/\n")
            f.write("develop-eggs/\n")
            f.write("dist/\n")
            f.write("downloads/\n")
            f.write("eggs/\n")
            f.write(".eggs/\n")
            f.write("lib/\n")
            f.write("lib64/\n")
            f.write("parts/\n")
            f.write("sdist/\n")
            f.write("var/\n")
            f.write("wheels/\n")
            f.write("*.egg-info/\n")
            f.write(".installed.cfg\n")
            f.write("*.egg\n")
            f.write("\n")
            f.write("# Virtual Environment\n")
            f.write("venv/\n")
            f.write("env/\n")
            f.write("ENV/\n")
            f.write("\n")
            f.write("# IDE\n")
            f.write(".idea/\n")
            f.write(".vscode/\n")
            f.write("*.swp\n")
            f.write("*.swo\n")
            f.write("\n")
            f.write("# Data\n")
            f.write("*.csv\n")
            f.write("*.xlsx\n")
            f.write("\n")
            f.write("# Output\n")
            f.write("output/\n")
        
        # Add and commit .gitignore
        subprocess.run(["git", "add", ".gitignore"], check=True)
        subprocess.run(["git", "commit", "-m", "Add .gitignore"], check=True)
    
    # Create GitHub repository
    if username:
        remote_url = f"https://github.com/{username}/{repo_name}.git"
        print(f"To push to GitHub, run the following commands:")
        print(f"  1. Create a new repository named '{repo_name}' on GitHub")
        print(f"  2. git remote add origin {remote_url}")
        print(f"  3. git branch -M main")
        print(f"  4. git push -u origin main")
    else:
        print(f"To push to GitHub, run the following commands:")
        print(f"  1. Create a new repository named '{repo_name}' on GitHub")
        print(f"  2. git remote add origin <repository-url>")
        print(f"  3. git branch -M main")
        print(f"  4. git push -u origin main")

def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(description="Set up a GitHub repository for the project")
    parser.add_argument('--repo_name', type=str, default="infertility-treatment-prediction", 
                        help='Name of the GitHub repository')
    parser.add_argument('--username', type=str, default=None,
                        help='GitHub username')
    
    args = parser.parse_args()
    setup_github(args.repo_name, args.username)

if __name__ == '__main__':
    main() 