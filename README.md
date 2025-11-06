# InformationRetrievalApplication
This is RAG based generative AI application where user can upload any pdf and query llm specific to the document.

## Installation and Setup

Follow these steps to install and run the application locally:

### 1. Clone the Project

First, clone the repository to your local machine:

```bash
git clone https://github.com/ramyaky/InformationRetrievalApplication.git
```

### 2. Open the Project in Your Favorite IDE

Navigate to the project folder and open it in your preferred Integrated Development Environment (IDE). For example, I recommend using VSCode

### 3. Create a Virtual Environment

To avoid cluttering your local Python environment with dependencies, it's best to use a virtual environment. Here's how to create one:

```bash
# Navigate to your project directory
cd InformationRetrievalApplication

# Create a virtual environment (use 'python3' if necessary)
python -m venv venv

# Activate the virtual environment
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```
### 4. Install the Required Dependencies

Install the dependencies listed in the requirements.txt file:

```bash
pip install -r requirements.txt
```
This will install all the necessary Python packages for the app to function properly.

### 5. Set Up Environment Variables

Before running the application, make sure to create a .env file in the root directory of your project. The .env file should contain your API keys as environment variables.

**1.** Create a .env file in the root directory of the project.

**2.** Add your **HUGGINGFACE_HUB_TOKEN** and **GOOGLE_API_KEY** to the .env file:
```bash
HUGGINGFACE_HUB_TOKEN=your-huggingface-token
GOOGLE_API_KEY=your-google-api-key
```
**Note:** The **HUGGINGFACE_HUB_TOKEN** is required to download the Google GEMMA embedding model locally from Hugging Face. Make sure you have a valid token to access the model.

### 6. Run the Application

Once your environment is set up, you can run the app with the following command:
```bash
streamlit run app.py
```
This will start a local development server, and you should be able to view the application by opening your browser and navigating to http://localhost:8501.






