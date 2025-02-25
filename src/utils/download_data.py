import os
import requests

def download_file_from_google_drive(file_id, destination):
    """Downloads a file from Google Drive using its file ID."""
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)
    print(f"Downloaded file to: {destination}")

def get_confirm_token(response):
    """Retrieves the confirmation token from cookies if present."""
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    """Saves the response content to a file in chunks."""
    CHUNK_SIZE = 32768
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  
                f.write(chunk)

if __name__ == "__main__":
    files_to_download = {
        "data/train_fraud_labels.json": "1sHZz2ZRDS55pq9e_9fsXl3G59QWBAduU",
        "data/transactions_data.csv": "1pIrqK2aMnfUTOpOVDFrML_WzHs5TP-St"
    }

    for destination, file_id in files_to_download.items():
        download_file_from_google_drive(file_id, destination)
