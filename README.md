# Search requests suggestion service with FastAPI

Use direct and reverse tries for searching of the best suggestion results.

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
