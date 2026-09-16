run:
	python run.py --data data --out predictions.csv

validate:
	python validate_submission.py predictions.csv

test:
	pytest -q
