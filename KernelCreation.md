# Creating the kernel


###### Download Python 3.12
winget install Python.Python.3.12


###### Create virtual environment from project directory
py -3.12 -m venv .venv


###### Activate environment
.\.venv\Scripts\Activate.ps1


###### Install requirements
python -m pip install -r requirements.txt


###### For Nvidia graphics cards
python -m pip uninstall torch torchvision torchaudio -y
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128


**Test with:**
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"


###### Create Jupyter kernel
python -m ipykernel install --user --name eurosat-env --display-name "Python (EuroSAT)"


###### In Powershell run
jupyter notebook

**OR**

jupyter lab