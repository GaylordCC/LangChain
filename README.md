# Create the virtual enviroment
python3 -m venv venv

# Active the virtual enviroment
source venv/bin/activate

# Python version of the project
Python 3.12.1

# Install the dependicies
pip install -r requirements.txt


# Run the server with: 
uvicorn Langchain.main:app --reload


# Open the project in the browser
localhost:8000
localhost:8000/docs