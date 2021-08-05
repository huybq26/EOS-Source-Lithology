## Installation

* Clone the repo
```
git clone https://github.com/taquynhnga2001/eos-lithology.git
cd eos-lithology
```

* Install virtualenv
```
pip install virtualenv
virtualenv env -p python3
```

* Install requirement in virtual env
```
source env/bin/activate
pip install -r requirements.txt
```

* Migrate 
```
python manage.py migrate
```

* Run the server
```
python manage.py runserver
```

The server is started at `localhost:8000`
