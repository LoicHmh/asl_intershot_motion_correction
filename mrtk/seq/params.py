import pandas as pd

class MRI:
    def __init__(self):
        
        data_dict = {
            "Tesla":        ["3T", "3T",  "3T", "7T", "7T",  "7T"],
            "Time_Type":    ["T1", "T2", "T2s", "T1", "T2", "T2s"],
            "WM":           [ 832,  110,  53.2, 1200,   47,  26.8],
            "GM":           [1331,   80,    66, 2000,   47,  33.2]
        }

        '''
        Source:
        https://aapm.onlinelibrary.wiley.com/doi/pdf/10.1002/mp.14986
        https://pubmed.ncbi.nlm.nih.gov/10232510
        '''

        self.data = pd.DataFrame(data_dict)

    def get_time(self, tesla: str, time_type: str, tissue: str) -> float:

        row = self.data[(self.data['Tesla'] == tesla) & (self.data['Time_Type'] == time_type)]
        if not row.empty:
            return row.iloc[0][tissue]
        else:
            return None