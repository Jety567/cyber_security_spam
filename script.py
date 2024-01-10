import subprocess
import os
import random
from tqdm import tqdm
import concurrent.futures

PERCENTAGE_OF_FILES = 0.7


def analyze_spamassassin_output(file_path):
    # Run the SpamAssassin command on the specified file
    process = subprocess.run(["spamassassin", file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Check the exit code and print the result
    result = process.stdout.decode("utf-8", errors='ignore')

    if "*****SPAM*****" in result:
        return True
    else:
        return False


def get_files_in_directory(directory_path):
    """
    Returns a list of all files in the specified directory.

    Parameters:
    - directory_path (str): The path of the directory.

    Returns:
    - List[str]: A list of file names in the directory.
    """
    try:
        # Use os.listdir to get the list of files in the directory
        files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
        return files
    except OSError as e:
        print(f"Error reading directory: {e}")
        return []


def prepare_array(files):
    filtered_files = [file for file in files if not file.endswith('.txt')]

    random.shuffle(filtered_files)

    length = PERCENTAGE_OF_FILES * len(filtered_files)

    length = round(length)

    filtered_files = filtered_files[:-length]

    return filtered_files


def spam_classification_loop(files, file_path, bar, data):
    for i in range(len(files)):
        data['spam'] += 1
        if analyze_spamassassin_output(f"~/spam_classification/{file_path}/{files[i]}"):
            data['spam_correct'] += 1
        bar.update(1)



if __name__ == "__main__":
    random.seed(42)
    data = {
        "ham": 0,
        "ham_correct": 0,
        "spam": 0,
        "spam_correct": 0
    }

    SPAM_FILE_PATH = "data/spam"
    HAM_FILE_PATH = "data/ham"

    spam_files = get_files_in_directory(SPAM_FILE_PATH)
    ham_files = get_files_in_directory(HAM_FILE_PATH)

    spam_files = prepare_array(spam_files)
    ham_files = prepare_array(ham_files)

    total_length = (len(spam_files) + len(ham_files))

    print(total_length)

    progress_bar = tqdm(total=total_length, desc='Progress', unit='iteration', dynamic_ncols=True)

    with concurrent.futures.ThreadPoolExecutor() as executor:

        params_spam = (spam_files, SPAM_FILE_PATH, progress_bar, data)
        params_ham = (ham_files, HAM_FILE_PATH, progress_bar, data)

        future_a = executor.submit(spam_classification_loop, *params_spam)
        future_b = executor.submit(spam_classification_loop, *params_ham)

        concurrent.futures.wait([future_a, future_b])

        print(data)
