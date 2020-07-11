#improve the speed, performance, etc of NN
import torch
import torch.utils.data as Data

BATCH_SIZE = 6
x = torch.linspace(1,10,10)
y = torch.linspace(10,1,10)

#torch_dataset = Data.TensorDataset(data_tensor = x, target_tensor = y)
torch_dataset = Data.TensorDataset(x, y)

#make trainning in batches
loader =Data.DataLoader(dataset = torch_dataset,
                       batch_size = BATCH_SIZE,
                       shuffle = True,
                       num_workers = 2) #shuffle is to decide whether disorder the data
                                        #num_work means number of process threads used
                                        
                                        
