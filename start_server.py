import subprocess
import sys
import os

def start_server():
    # Check if model file exists
    if not os.path.exists("cnn_pipeline_model.zip"):
        print("❌ Error: cnn_pipeline_model.zip not found!")
        print("Please ensure the model file is in the project directory")
        return
    
    print("🚀 Starting Flask server...")
    try:
        subprocess.run([sys.executable, "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    start_server()