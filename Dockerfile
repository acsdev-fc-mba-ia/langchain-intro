# Use the official Python 3 image based on Alpine Linux
FROM mcr.microsoft.com/devcontainers/python:3.12-bullseye

    # Create and activate a virtual environment
# Do not need to do this in a docker file, because I can consider the container as an isolated environment
# RUN python3 -m venv venv &&. venv/bin/activate 

RUN apt update && apt install -y git-lfs

# PYTHON LIBS
RUN pip install langchain langchain-openai langchain-google-genai \
    langchain-community langchain-text-splitters \
    langchain-postgres psycopg[binary] python-dotenv beautifulsoup4 pypdf