# JobStalker | Telegram Vacancy Filter Bot

<p align="center"> <img src="job-stalker/src/job_stalker/static-files/images/Banner.png" alt="Banner" width="1200"/> </p> <p align="center"> <a href="https://pypi.org/project/job-stalker/"> <img src="https://img.shields.io/pypi/v/job-stalker" alt="PyPI version"/> </a> </p>

This is a Telegram bot that filters job vacancies from specified channels using a machine learning model. It has a web interface for configuration, authorization, and viewing results.

## ✨ Features

-   Filters vacancies from public Telegram channels.
-   Uses a machine learning model to determine the relevance of a vacancy.
-   Web interface for easy configuration and authorization in Telegram.
-   Real-time monitoring of new vacancies.
-   Saves results in a database.

## 🖼 Screenshots

<div style="display: flex; overflow-x: auto; padding: 10px 0;">
  <img src="job-stalker/src/job_stalker/static-files/images/1.png" width="200" style="margin-right:10px;"/>
  <img src="job-stalker/src/job_stalker/static-files/images/2.png" width="200" style="margin-right:10px;"/>
  <img src="job-stalker/src/job_stalker/static-files/images/3.png" width="200" style="margin-right:10px;"/>
  <img src="job-stalker/src/job_stalker/static-files/images/4.png" width="200" style="margin-right:10px;"/>
  <img src="job-stalker/src/job_stalker/static-files/images/5.png" width="200" style="margin-right:10px;"/>
</div>

## 🚀 Installation and Usage

> **⚠️ Important!**
> This application is a Telegram client that works on behalf of your personal account. You **must** obtain your own `API_ID` and `API_HASH` from [my.telegram.org](https://my.telegram.org).
>
> **Do not share your API keys with anyone.** Using someone else's keys or sharing yours is a violation of Telegram's terms of service and can lead to your account being banned.

There are four ways to run this application:

1.  [Install via pip (Recommended)](#install-via-pip)
2.  [Using Docker ](#using-docker)
3.  [From source code](#from-source-code)
4.  [As a standalone executable](#standalone-executable)

### 1️⃣ Install via pip

Use this option if you don't need the source code and just want to start quickly.

**Prerequisites:**

-   Python 3.11+ and `pip`.

**Steps:**

1.  **Create a working directory:**
    ```bash
    mkdir job-stalker-data
    cd job-stalker-data
    ```

2.  **Install the package:**
    ```bash
    pip install job-stalker       # for CPU
    pip install "job-stalker[gpu]"  # or just job-stalker if you don't need GPU monitoring (optional)
    ```

3.  **Download and configure `.env`:**
    ```bash
    curl -o .env.example https://raw.githubusercontent.com/10sorry/JobStalker/main/.env.example
    cp .env.example .env
    # fill in API_ID, API_HASH and optional proxy settings
    ```

4.  **Run the web interface:**
    ```bash
    job-stalker
    ```
    The server will start on `http://0.0.0.0:8000`. Run the command from the directory that contains `.env`.

### 2️⃣ Using Docker

This is the easiest and recommended way to run the application.

**Prerequisites:**

-   [Docker](https://docs.docker.com/get-docker/) installed on your system.
-   [Docker Compose](https://docs.docker.com/compose/install/) installed on your system.

**Steps:**

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd telegram-filter-bot
    ```

2.  **Create the environment file:**
    Copy the example `.env.example` file to `.env` and fill in your details.
    ```bash
    cp .env.example .env
    ```
    You will need to get your `API_ID` and `API_HASH` from [my.telegram.org](https://my.telegram.org).

3.  **Build and run the container:**
    ```bash
    docker-compose up --build
    ```
    This will build the Docker image and start the application.

4.  **Open the web interface:**
    Open your web browser and go to `http://localhost:8000`.

5.  **Authorize and configure:**
    -   You will be prompted to authorize your Telegram account.
    -   In the settings, specify the channels you want to monitor and other parameters.

6.  **Start scanning:**
    Click the "START SCAN" button to begin the filtering process.

### 3️⃣ From Source Code

This method is for users who want to run the application directly from the source code.

**Prerequisites:**

-   Python 3.11 or higher.
-   `venv` module for creating virtual environments.

**Steps:**

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd telegram-filter-bot
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
    On Windows, use `venv\Scripts\activate`.

3.  **Install the dependencies (editable mode):**
    ```bash
    pip install -e ".[gpu]"   # or without [gpu] if you don't need GPU monitoring
    ```

4.  **Create the environment file:**
    Copy the example `.env.example` file to `.env` and fill in your details.
    ```bash
    cp .env.example .env
    ```

5.  **Run the application:**
    ```bash
    job-stalker
    ```
    (You can also use `python -m job_stalker.run`, but the CLI script is shorter.) This will start the web server.

6.  **Open the web interface:**
    Open your web browser and go to `http://localhost:8000` and follow the configuration steps as described in the Docker section.

### 4️⃣ Standalone Executable

This method is for users who want to run the application without installing Python or any dependencies.

**Steps:**

1.  **Download the executable:**
    Go to the [Releases](<repository-url>/releases) page and download the `telegram_filter_bot` executable for your operating system.

2.  **Create the environment file:**
    Create a `.env` file in the same directory as the executable and fill in your `API_ID` and `API_HASH`.
    ```
    API_ID=...
    API_HASH=...
    # Optional proxy settings
    # PROXY_ENABLED=false
    # PROXY_HOST=
    # PROXY_PORT=
    # PROXY_USER=
    # PROXY_PASS=
    ```

3.  **Run the executable:**
    On Linux and macOS, you may need to make the file executable first:
    ```bash
    chmod +x telegram_filter_bot
    ```
    Then, run the application:
    ```bash
    ./telegram_filter_bot
    ```
    On Windows, you can just double-click the `.exe` file.

4.  **Open the web interface:**
    Open your web browser and go to `http://localhost:8000`.

## 🛠 Development

If you want to contribute to the project, you can follow the instructions for running from source code.

### Building the pip package

Artifacts for PyPI/TestPyPI are built from `pyproject.toml`.

```bash
pip install build twine
python -m build            # produces dist/*.whl and dist/*.tar.gz
twine upload --repository testpypi dist/*
# after verification:
twine upload dist/*
```

### Building the Executable

To build the executable yourself, you need to have Python and the project dependencies installed. Then, run the build script:

```bash
python pyinstaller_build.py
```
The executable will be created in the `dist` directory.

### ⚙️ .gitignore

The following files and directories are ignored by Git:

```
.env
/data/
/venv/
/build/
/dist/
*.spec
__pycache__/
*.pyc
```

---
