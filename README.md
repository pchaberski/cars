# cars

TODO: a proper README

Experiment tracking:
https://ui.neptune.ai/pchaberski/cars/experiments?viewId=139ea15a-c997-4779-8abb-c2b937db28d6

Setting up venv and installing PyTorch with pip (on Windows):

prod_requirements.txt:

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

(cars) C:\Users\pchaberski\Google Drive\gdprojects\cars>pip install -r prod_requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```
