# EOS-Source-Lithology

Please install requirements package found in requirements.txt beforehand.

Run in production mode:

### `flask run` 

Run in debug mode: 

### `python app.py` 

Deployment note:

The default excel reading engine is not openpyxl but something else. Hence, need to specify the engine in each read_excel function.
