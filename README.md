# cars

Assumption:
Data is downloadeded from [https://ai.stanford.edu/~jkrause/cars/car_dataset.html](https://ai.stanford.edu/~jkrause/cars/car_dataset.html).
All 3 components:  

- `cars_train` (folder containing training images)
- `cars_test` (folder containing validation images)
- `car_devkit` (collection of files containing among others image labels)

are unzipped and stored in the same folder, which location in provided in `config.yml` (`data_path` parameter) without any modifications:

```
C:\Users\pchaberski\Google Drive\gdprojects\cars\input\stanford>tree /A
Folder PATH listing for volume Windows
Volume serial number is CE5C-2E3D
C:.
+---cars_test
+---cars_train
\---car_devkit
    \---devkit

C:\Users\pchaberski\Google Drive\gdprojects\cars\input\stanford>tree /F car_devkit\devkit
Folder PATH listing for volume Windows
Volume serial number is CE5C-2E3D
C:\USERS\PCHABERSKI\GOOGLE DRIVE\GDPROJECTS\CARS\INPUT\STANFORD\CAR_DEVKIT\DEVKIT
    cars_meta.mat
    cars_test_annos.mat
    cars_train_annos.mat
    eval_train.m
    README.txt
    train_perfect_preds.txt

No subfolders exist
```

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

(cars) C:\Users\pchaberski\Google Drive\gdprojects\cars>pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```
