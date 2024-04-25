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