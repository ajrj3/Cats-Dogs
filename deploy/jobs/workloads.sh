# set up aws account credentials via environment variables
export AWS_ACCESS_KEY_ID="REDACTED"
export AWS_SECRET_ACCESS_KEY="REDACTED"

source /home/user/.virtualenvs/catsdogs_env/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD

export RESULTS_FILE=results/test_data_results.txt
pytest tests/test_data.py --disable-warnings > $RESULTS_FILE
cat $RESULTS_FILE

export RESULTS_FILE=results/test_utils_results.txt
pytest tests/test_utils.py --disable-warnings > $RESULTS_FILE
cat $RESULTS_FILE

export RESULTS_FILE=results/test_model_results.txt
pytest tests/test_model_fns.py --disable-warnings > $RESULTS_FILE
cat $RESULTS_FILE

export RESULTS_FILE=results/train_results.json
python catsdogs/model_fns.py train-model-distributed --results-filepath $RESULTS_FILE --epochs 4 
cat $RESULTS_FILE

# upload artefacts to s3 bucket
aws s3 cp results/ s3://cats-dogs-classifier --recursive