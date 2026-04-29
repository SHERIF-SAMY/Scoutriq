#!/bin/bash
# Startup script for Google Compute Engine VM
# Runs automatically as root on boot

# 1. Update packages
apt-get update -y
apt-get upgrade -y

# 2. Install Python, pip, and system dependencies for OpenCV/Mediapipe
apt-get install -y python3 python3-pip python3-venv git
apt-get install -y ffmpeg libsm6 libxext6 libgl1-mesa-glx

# 3. Create a directory for the app
APP_DIR="/opt/scoutai"
mkdir -p $APP_DIR
cd $APP_DIR

# 4. We will deploy the code here. In a real scenario, you usually git clone your repo here.
# Since we don't know the exact github url, we will just echo a manual setup instruction
# FOR NOW: Let's assume you've already zipped the local scoutai folder, and put it on Google Cloud Storage.
# If you didn't do that, you'll need to manually `scp` (Secure Copy) your local scoutai files to this VM in /opt/scoutai.
# (Alternative: you can push to github, and `git clone <YOUR_REPO>` here)

echo "Please upload your scoutai files to $APP_DIR" > /opt/scoutai/README.txt

# 5. [Assuming files are present] Install dependencies to the global system path for simplicity 
# (Or setup a virtualenv: python3 -m venv venv && source venv/bin/activate)
pip3 install virtualenv
virtualenv venv
source venv/bin/activate

# Wait for code...
# Assuming we run this manually or code is pulled via git:
# pip install -r requirements.txt

# Create a systemd service to keep the API running automatically
cat <<EOF > /etc/systemd/system/scoutai.service
[Unit]
Description=ScoutAI FastAPI Service
After=network.target

[Service]
User=root
Group=root
WorkingDirectory=/opt/scoutai
Environment="PATH=/opt/scoutai/venv/bin"
# Executable will fail if api.py is not yet populated via git/scp, but will auto-restart
ExecStart=/opt/scoutai/venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Reload and Start the service
systemctl daemon-reload
systemctl enable scoutai.service
systemctl start scoutai.service

echo "Startup script finished."
