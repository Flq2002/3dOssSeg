class Config:
    def __init__(self,task):
        if "la" in task: # SSL
            self.base_dir = '/media/HDD/fanlinqian/LASeg'
            self.save_dir = '/media/HDD/fanlinqian/LASeg/LA_process'
            # self.patch_size = (112, 112, 80)
            self.patch_size = (128, 128, 64)
            # self.patch_size = (256,160,96)
            self.num_cls = 2
            self.num_channels = 1
            self.n_filters = 32
            self.early_stop_patience = 100
            self.batch_size = 1
            self.norm_cfg = {'mean':0.8892374, 'std': 1.0632252}
        elif "lv" in task: # SSL
            self.base_dir = '/media/HDD/fanlinqian/liver'
            self.save_dir = '/media/HDD/fanlinqian/liver'
            self.patch_size = (128,128,128)
            self.num_cls = 3
            self.num_channels = 1
            self.n_filters = 32
            self.early_stop_patience = 100
            self.batch_size = 2
        elif "synapse" in task: # IBSSL
            self.base_dir = './Datasets/Synapse'
            self.save_dir = './Synapse_data'
            self.patch_size = (64, 128, 128)
            self.num_cls = 14
            self.num_channels = 1
            self.n_filters = 32
            self.early_stop_patience = 100
            self.batch_size = 2

        elif "mmwhs" in task: # UDA
            self.base_dir = '/home/HDD/fanlinqian/MMWHS/Dataset'
            self.save_dir = '/home/HDD/fanlinqian/MMWHS'
            self.patch_size = (128, 128, 128)
            self.num_cls = 5
            self.num_channels = 1
            self.n_filters = 32
            self.early_stop_patience = 100
            self.batch_size = 2

        elif "mnms" in task: # SemiDG
            if "2d" in task:
                self.base_dir = './Datasets/MNMs/Labeled'
                self.save_dir = './MNMS_data_2d'
                self.patch_size = (224, 224)
                self.num_cls = 4
                self.num_channels = 1
                self.n_filters = 32
                self.early_stop_patience = 50
                self.batch_size = 32
            else:
                self.base_dir = './Datasets/MNMs/Labeled'
                self.save_dir = './MNMS_data'
                self.patch_size = (32, 128, 128)
                self.num_cls = 4
                self.num_channels = 1
                self.n_filters = 32
                self.early_stop_patience = 80
                self.batch_size = 4
        elif "oss1" in task:
            self.base_dir = '/media/HDD/fanlinqian/ossicular/data1'
            self.save_dir = '/media/HDD/fanlinqian/ossicular/data1_process'
            self.patch_size = (640, 640, 360)
            self.num_cls = 2
            self.num_channels = 1
            self.n_filters = 32
            self.early_stop_patience = 50
            self.batch_size = 1
        elif "oss2" in task:
            self.base_dir = '/media/HDD/fanlinqian/ossicular/data2'
            self.save_dir = '/media/HDD/fanlinqian/ossicular/data2_process'
            self.patch_size = (640, 640, 360)
            self.num_cls = 2
            self.num_channels = 1
            self.n_filters = 32
            self.early_stop_patience = 50
            self.batch_size = 1
        elif "oss3" in task:
            self.base_dir = '/media/HDD/fanlinqian/ossicular/data3_253'
            self.save_dir = '/media/HDD/fanlinqian/ossicular/data3_253'
            # self.patch_size = (640, 640, 360)
            self.num_cls = 2
            self.num_channels = 1
            self.n_filters = 32
            self.early_stop_patience = 50
            self.batch_size = 1
        elif "oss4" in task:
            self.base_dir = '/media/HDD/fanlinqian/ossicular/data4'
            self.save_dir = '/media/HDD/fanlinqian/ossicular/data4'
            # self.patch_size = (640, 640, 360)
            self.num_cls = 2
            self.num_channels = 1
            self.n_filters = 32
            self.early_stop_patience = 50
            self.batch_size = 1
        else:
            raise NameError("Please provide correct task name, see config.py")


