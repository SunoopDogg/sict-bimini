from utils.change_extension import bim_xlsx_to_json
from utils.xlsx_list import load_filenames_from_txt


if __name__ == '__main__':
    file_names = load_filenames_from_txt()

    for file_name in file_names:
        json_data = bim_xlsx_to_json(file_name)
