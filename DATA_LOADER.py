import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import torch



class CustomDataset(Dataset):
    def __init__(self, input_folder, target_folder):
        self.folder_path = input_folder
        self.target_path = target_folder

        self.input_file_list = os.listdir(input_folder)
        self.target_file_list = os.listdir(target_folder)




    def __len__(self):
        return len(self.input_file_list)





    def __getitem__(self, idx):
        input_file_name = self.input_file_list[idx]
        input_file_id = input_file_name.split('_')[-1].split('.')[0]

        # Sprawdzamy, czy istnieje odpowiadający plik docelowy
        target_file_name = f'TARGET_{input_file_id}.npy'
        target_file_path = os.path.join(self.target_path, target_file_name)

        if os.path.exists(target_file_path):
            input_file_path = os.path.join(self.folder_path, input_file_name)
            try:
                input_data = np.load(input_file_path).astype(np.float32)
                target_data = np.load(target_file_path).astype(np.float32)

                # if input_data.dtype != np.float32:
                #     # Jeśli typ danych nie jest float32, konwertuj na float32
                #     input_data = input_data.astype(np.float32)


                input_tensor = torch.tensor(input_data[10, :, :, :], dtype=torch.float32)
                target_tensor = torch.tensor(target_data[0, :, :], dtype=torch.float32)

                return input_tensor, target_tensor
            except TypeError as e:
                # Jeśli wystąpił błąd TypeError, wypisz nazwę pliku, który spowodował błąd
                print(f"Błąd w przetwarzaniu pliku: {input_file_name}, Error: {str(e)}")
                return None, None
        else:
            # Jeśli plik docelowy nie istnieje, zwracamy None w celu pominięcia tego przykładu
            return None, None



















def collate_fn(batch):
    # Usuwamy None z listy batch
    batch = [x for x in batch if x[0] is not None]

    if len(batch) == 0:
        # Jeśli wszystkie pliki były puste (brak odpowiadających plików docelowych), zwracamy pustą listę
        return [], []

    # Sortujemy batch względem długości sekwencji w tensorze input
    batch = sorted(batch, key=lambda x: x[0].size(2), reverse=True)

    # Rozszerzenie danych docelowych o wymiar channels
    target_data = torch.stack([x[1] for x in batch])

    # Zamiana kolejności kanałów i klatek czasowych w danych wejściowych
    input_data = torch.stack([x[0] for x in batch])

    return input_data, target_data