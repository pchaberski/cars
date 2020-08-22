# cars

setting up venv and installing pytorch with pip:

requirements.txt:

```
...
torch==1.6.0
torchvision==0.7.0
...
```

run using `cmd`:

```
C:\Users\pchaberski>cd C:\Users\pchaberski\Google Drive\gdprojects\cars

C:\Users\pchaberski\Google Drive\gdprojects\cars>python -m venv C:\projects\venvs\cars

C:\Users\pchaberski\Google Drive\gdprojects\cars>C:\projects\venvs\cars\Scripts\activate.bat

(cars) C:\Users\pchaberski\Google Drive\gdprojects\cars>pip install -r dev_requirements.txt

(cars) C:\Users\pchaberski\Google Drive\gdprojects\cars>pip install -r prod_requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```
