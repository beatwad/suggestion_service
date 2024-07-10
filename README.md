# Search requests suggestion service with FastAPI

## Service start

`bash run.sh`

Default address: http://localhost:8000/

## Tests

`pytest .`

Test run with the verbose output (errors и prints):

`pytest -svv .`

Run one test:

`pytest -svv tests/test_app.py::test_suggestions_recall`

## Pylint

`pylint app.py`
