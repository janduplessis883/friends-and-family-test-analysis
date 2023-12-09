install:
	@pip install -e .

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -rf */.ipynb_checkpoints
	@rm -Rf build
	@rm -Rf */__pycache__
	@rm -Rf */*.pyc
	@echo "🧽 Cleaned up successfully!"

all: install clean

test:
	@pytest -v tests

app:
	@streamlit run friendsfamilytest/app.py

app2:
	@streamlit run friendsfamilytest/app2.py

gsheet:
	@streamlit run friendsfamilytest/gsheet_connect.py

data:
	@python friendsfamilytest/data.py

git_push:
	@python friendsfamilytest/auto_git/git_push.py

# Specify package name
lint:
	@black friendsfamilytest/
