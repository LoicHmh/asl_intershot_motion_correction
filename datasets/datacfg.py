import os
import tqdm
import mapvbvd


def check_single_twix(file_path: str):
    try:
        twixObj: mapvbvd.mapVBVD = mapvbvd.mapVBVD(file_path, quiet=True)
        return True
    except:
        print(f'Error: {file_path} is not a valid twix file')
        return False
    

class RawData():
    def __init__(self, twix_path: str, online_recon_nii_path: str = '', protocol: str = '', alias: str = '', data_dir: str = '', recon_settings: dict = {}) -> None:
        self.data_dir = data_dir
        self.twix_path = os.path.join(data_dir, twix_path)
        self.online_recon_nii_path = os.path.join(data_dir, online_recon_nii_path)
        self.protocol = protocol
        self.alias = alias
        self.recon_settings = recon_settings
        self.name = self.get_name()

    def check_integrity(self):
        return check_single_twix(self.twix_path)
    
    def __str__(self) -> str:
        return f'RawData: {self.alias}, protocol: {self.protocol}\ntwix_path: {self.twix_path}\nonline_recon_nii_path: {self.online_recon_nii_path}'
    
    def get_name(self):
        # Get the name of the file without the extension
        return os.path.splitext(os.path.basename(self.twix_path))[0]



class Experiment():
    def __init__(self, subject: str, magnetic_field: str, date: str, data_root: str, exp_name: str, description: str) -> None:
        self.subject = subject
        self.magnetic_field = magnetic_field
        self.date = date
        self.raw_data_list = []
        self.data_root = data_root
        self.exp_name = exp_name
        self.data_dir = os.path.join(self.data_root, self.exp_name)
        self.description = description

    def check_integrity(self):
        counter = 0
        for raw_data in tqdm.tqdm(self.raw_data_list, total=len(self.raw_data_list)):
            if raw_data.check_integrity():
                counter += 1

        print(f'Checked {len(self.raw_data_list)} raw data files, {counter} good.')
        return counter == len(self.raw_data_list)
    
    def add_raw_data(self, raw_data: RawData):
        self.raw_data_list.append(raw_data)

    def check_names(self):
        names = []
        for raw_data in self.raw_data_list:
            if raw_data.name in names:
                print(f'Error: {raw_data.name} is not unique')
            else:
                names.append(raw_data.name)

    def __call__(self, alias: str = ''):

        for raw_data in self.raw_data_list:
            if raw_data.alias == alias:
                return raw_data
        return None

    def __str__(self) -> str:
        return f'Experiment: {self.exp_name} on {self.date} for {self.subject} at {self.magnetic_field}\nDescription: {self.description}'

