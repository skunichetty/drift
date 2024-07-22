from libextract import run_extract
from libextract.export import StorageMode
import os


def lambda_handler(event, context):
    run_extract(event["url"], StorageMode.SQL, None, os.environ["DB_URI"], verbose=True)
