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

app3:
	@streamlit run friendsfamilytest/app_test.py

app_wide:
	@streamlit run friendsfamilytest/app_wide.py

gsheet:
	@streamlit run friendsfamilytest/gsheet_connect.py

data:
	@python friendsfamilytest/data.py

repeat:
	@python friendsfamilytest/scheduler.py

git_merge:
	$(MAKE) clean
	@python friendsfamilytest/auto_git/git_merge.py

git_push:
	$(MAKE) clean
	$(MAKE) lint
	@python friendsfamilytest/auto_git/git_push.py


lint:
	@black friendsfamilytest/

