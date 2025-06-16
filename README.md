## 1. Install Visual Studio Code (VS Code)

Download and install VS Code from the official site:
https://code.visualstudio.com/

---

## 2. Install Conda (Anaconda or Miniconda)

You can use either **Anaconda** or **Miniconda**. There is a tutorial on Moodle howto setup environments.

- **Anaconda** (includes many preinstalled packages):  
https://www.anaconda.com/products/distribution

- **Miniconda** (minimal version, lighter):  
https://docs.conda.io/en/latest/miniconda.html

---

## 3. Install Required VS Code Extensions

Open VS Code, go to the **Extensions** tab and install the following extensions:

- **Python**
- **Code Runner** (for running the code)  

---

## 4. Clone the Repository

git clone https://github.com/shreyasdesikan/Dynamic-Process-Model
cd Dynamic-Process-Model

Make sure the environment.yaml file is in this directory.

---

## 5. Create Conda Environment from environment.yaml

Run the following command inside the cloned repo directory:

conda env create -f environment.yaml

This will create a new environment with the required dependencies.

---

## 6. Select Conda Environment in VS Code

To use the correct Python interpreter in VS Code:

Open VS Code in the project folder.
Open the Command Palette (Ctrl+Shift+P or Cmd+Shift+P).
Search and select "Python: Select Interpreter".
Choose the interpreter corresponding to your activated Conda environment.


