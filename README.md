# Elderly-ai-companion
Emotion Aware LLM for Elderly Support

# Project Setup Guide
This guide will walk you through setting up your development environment for this project.

1. Create a Python Virtual Environment
It's highly recommended to use a virtual environment to manage project dependencies.

## Create a virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

## 2. Download and Set Up Ollama
Ollama is required to run local language models.

Download Ollama: Visit the official Ollama website and download the appropriate installer for your operating system:
https://ollama.com/download

Install Ollama: Follow the installation instructions provided on the Ollama website or by the installer.

Verify Installation: Open your terminal or command prompt and run:

ollama --version

You should see the installed Ollama version.

## 3. Pull the gemma3:4b Model
Once Ollama is installed, you can pull the gemma3:4b model.

ollama pull gemma3:4b

This command will download the model to your local machine.

## 4. Download Project Data from Google Drive
A folder containing necessary project data needs to be downloaded from Google Drive.

Open the Google Drive link:
https://drive.google.com/drive/folders/12D2nNK2ekL58PGqzW0pKYxzCJjq0a-LT?usp=drive_link

Download the folder: Download the entire folder to your local machine.

Place the folder in the project directory: After downloading, move the entire downloaded folder into the root directory of this project. Ensure the folder structure matches what the project expects.

You are now ready to start working on the project!
