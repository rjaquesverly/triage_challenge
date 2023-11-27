# Triage Challenge Script - Team name : MGB-Harvard

This guide explains how to set up and run the `triage_challenge.py` script.

## Prerequisites
Ensure you have Python installed on your system.
This code and model was developed using Python 3.9.17.
You can download the Python 3.9.17. using this link: https://www.python.org/ftp/python/3.12.0/python-3.12.0b1-amd64.exe

## Installation

1. ### Clone the Repository:

   First, clone the repository to your local machine:
   
   ```bash
   git clone https://github.com/rjaquesverly/triage_challenge.git
   ```

2. ### Access the triage_challenge folder in your computer:

   ```bash
   cd triage_challenge
   ```

3. ### Install Required Packages:

   ```bash
   python -m pip install --upgrade pip
   pip install numpy scikit-learn tensorflow
   ```
4. ### Running the Script
   ```bash
   python triage_challenge.py
   ```
## Additional Notes
Ensure that the data files csv files are inside the data folder.
The script uses TensorFlow and may require a significant amount of RAM and a powerful CPU or GPU.
The output will be saved in a JSON format as specified in the script.


