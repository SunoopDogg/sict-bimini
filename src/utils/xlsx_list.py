import os


def get_xlsx_filenames(directory: str):
    """
    주어진 디렉토리에서 xlsx 파일의 이름을 추출하는 함수

    Args:
        directory (str): 디렉토리 경로

    Returns:
        list: xlsx 파일 이름 리스트
    """
    xlsx_files = [os.path.splitext(f)[0] for f in os.listdir(
        directory) if f.endswith('.xlsx')]

    return xlsx_files


def save_filenames_to_txt(filenames: list, file_path: str):
    """
    파일 이름 리스트를 텍스트 파일로 저장하는 함수

    Args:
        filenames (list): 파일 이름 리스트
        file_path (str): 저장할 텍스트 파일 경로
    """
    with open(file_path, 'w') as file:
        for name in filenames:
            file.write(f"{name}\n")


def load_filenames_from_txt(file_path="data/txt/xlsx_filenames.txt"):
    """
    텍스트 파일에서 파일 이름 리스트를 불러오는 함수

    Args:
        file_path (str): 텍스트 파일 경로

    Returns:
        list: 파일 이름 리스트
    """
    with open(file_path, 'r') as file:
        filenames = [line.strip() for line in file.readlines()]
    return filenames


# data/xlsx 폴더 안에 있는 xlsx 파일 이름을 추출
xlsx_directory = os.path.join(os.getcwd(), "data", "xlsx")
xlsx_list = get_xlsx_filenames(xlsx_directory)

# 추출한 파일 이름들을 data/txt 폴더 안에 저장
txt_directory = os.path.join(os.getcwd(), "data", "txt")
os.makedirs(txt_directory, exist_ok=True)
txt_file_path = os.path.join(txt_directory, "xlsx_filenames.txt")
save_filenames_to_txt(xlsx_list, txt_file_path)
