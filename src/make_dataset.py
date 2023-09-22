import requests
import os
import pandas as pd


def load_dataset() -> pd.DataFrame:
    """
    Loads the dataset and downloads the dataset if it doesn't exists
    Returns
    """
    file_path = "./data/product-rating.csv"

    if not os.path.exists(file_path):
        print("Downloading dataset....")
        __download_file_from_google_drive(file_path)

    print("Loading dataset...")
    df = pd.read_csv(file_path)[:10000]
    print("Dataset loaded")
    df[['category', 'subcategory']] = tuple(df['category_code'].apply(__extract_categorycode))
    
    return df


def __extract_categorycode(input_text):
    """
     This function splits category code into category and subcategory.
    """
    
    if str(input_text) == "nan": return (None, None)
    output_text = input_text.split('.')
    output = (output_text[0], output_text[1]+("."+output_text[2] if len(output_text) == 3 else ""))

    return output


def __get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def __save_response_content(response, file_path):
    CHUNK_SIZE = 32768

    with open(file_path, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: 
                f.write(chunk)


def __download_file_from_google_drive(file_path):
    URL = "https://docs.google.com/uc?export=download"
    file_id = os.getenv("file_id")

    session = requests.Session()

    response = session.get(URL, params = { 'id' : file_id }, stream = True)
    token = __get_confirm_token(response)

    if token:
        params = { 'id' : file_id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    __save_response_content(response, file_path)