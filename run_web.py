"""
FoodMind Web UI - start script
Run from the project root:  python run_web.py
"""
import subprocess, sys, os

def main():
    web_app = os.path.join(os.path.dirname(__file__), "web", "app.py")
    print("\nFoodMind - AI Food Recommendation System")
    print("=" * 50)
    print("  Starting web server...")
    print("  Open -> http://localhost:5000")
    print("  Press CTRL+C to stop\n")
    subprocess.run([sys.executable, web_app])

if __name__ == "__main__":
    main()
