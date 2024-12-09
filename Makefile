build:
	pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
	pip -r install requirements.txt
run:
	uvicorn app:app --reload --host 0.0.0.0 --port 3000

clean:
	pip uninstall -y torch torchvision fastapi uvicorn python-multipart scikit-learn pillow transformers pandas
	rm -rf __pycache__ *.pyc *.pyo