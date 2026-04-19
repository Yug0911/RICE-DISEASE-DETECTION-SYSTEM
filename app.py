"""
Entry point for Hugging Face Spaces deployment
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from demo_extraordinary import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 1000))
    app.run(host="0.0.0.0", port=port, debug=False)