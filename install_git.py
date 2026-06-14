import urllib.request
import zipfile
from pathlib import Path
import sys

def main():
    print("[MinGit Installer] Initializing portable Git setup...")
    git_dir = Path("C:/Users/kehgy/.gemini/antigravity/scratch/git")
    git_dir.mkdir(parents=True, exist_ok=True)
    
    # If already installed, skip
    if (git_dir / "cmd" / "git.exe").exists():
        print(f"[MinGit Installer] MinGit is already set up at: {git_dir}")
        return
    
    # 1. Fetch latest release version strings
    try:
        print("[MinGit Installer] Fetching latest version tags...")
        req_tag = urllib.request.Request("https://gitforwindows.org/latest-tag.txt", headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req_tag, timeout=10) as response:
            tag = response.read().decode().strip()
            
        req_ver = urllib.request.Request("https://gitforwindows.org/latest-version.txt", headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req_ver, timeout=10) as response:
            version = response.read().decode().strip()
    except Exception as e:
        print(f"[MinGit Installer] Warning: could not resolve latest version dynamically: {e}")
        # Fallback to a verified stable version if network tags fail
        tag = "v2.45.1.windows.1"
        version = "2.45.1"
        
    url = f"https://github.com/git-for-windows/git/releases/download/{tag}/MinGit-{version}-64-bit.zip"
    zip_path = Path("C:/Users/kehgy/.gemini/antigravity/scratch/mingit.zip")
    
    print(f"[MinGit Installer] Downloading MinGit from {url}...")
    try:
        # Fetching ZIP file
        req_dl = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req_dl, timeout=30) as response:
            with open(zip_path, 'wb') as out_file:
                out_file.write(response.read())
                
        print("[MinGit Installer] Extracting MinGit ZIP archive...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(git_dir)
            
        print(f"[MinGit Installer] MinGit set up successfully at: {git_dir}")
        
        # Cleanup zip file
        if zip_path.exists():
            zip_path.unlink()
            
    except Exception as e:
        print(f"[MinGit Installer] Error installing MinGit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
