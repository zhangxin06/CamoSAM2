from .btcv import BTCV
from .amos import AMOS
from .vcod import VCOD, VCOD_new, VCOD_Train_EPFlow, VCOD_Test_EPFlow, VCOD_all_train
from .cod import COD
import torchvision.transforms as transforms
from torch.utils.data import DataLoader



def get_dataloader(args, test_data_name=None):
    transform_train = transforms.Compose([
        transforms.Resize((args.image_size,args.image_size)),
        transforms.ToTensor(),
    ])

    transform_train_seg = transforms.Compose([
        transforms.Resize((args.out_size,args.out_size)),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    transform_test_seg = transforms.Compose([
        transforms.Resize((args.out_size,args.out_size)),
        transforms.ToTensor(),
    ])
    
    if args.dataset == 'btcv':
        '''btcv data'''
        btcv_train_dataset = BTCV(args, args.data_path, transform = transform_train, transform_msk= transform_train_seg, mode = 'Training', prompt=args.prompt)
        btcv_test_dataset = BTCV(args, args.data_path, transform = transform_test, transform_msk= transform_test_seg, mode = 'Test', prompt=args.prompt)

        nice_train_loader = DataLoader(btcv_train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(btcv_test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        '''end'''
    elif args.dataset == 'amos':
        '''amos data'''
        amos_train_dataset = AMOS(args, args.data_path, transform = transform_train, transform_msk= transform_train_seg, mode = 'Training', prompt=args.prompt)
        amos_test_dataset = AMOS(args, args.data_path, transform = transform_test, transform_msk= transform_test_seg, mode = 'Test', prompt=args.prompt)

        nice_train_loader = DataLoader(amos_train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(amos_test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        '''end'''
    elif args.dataset == 'vcod':
        '''vcod data'''
        vcod_train_dataset = VCOD(args, args.data_path, transform = transform_train, transform_msk= transform_train_seg, mode = 'TrainDataset_per_sq', prompt=args.prompt)
        vcod_test_dataset = VCOD(args, args.data_path, transform = transform_test, transform_msk= transform_test_seg, mode = 'TestDataset_per_sq', prompt=args.prompt) # MoCA
        # vcod_test_dataset = VCOD(args, args.data_path, transform = transform_test, transform_msk= transform_test_seg, mode = 'CAD', prompt=args.prompt) # CAD



        # vcod_train_dataset = VCOD_all_train(args, args.data_path, transform = transform_train, transform_msk= transform_train_seg, mode = 'TrainDataset_per_sq', prompt=args.prompt)
        # vcod_test_dataset = VCOD_new(args, args.data_path, transform = transform_test, transform_msk= transform_test_seg, mode = 'TestDataset_per_sq', prompt=args.prompt)

        # vcod_train_dataset = VCOD_Train_EPFlow(args, args.data_path, transform=transform_train, transform_msk=transform_train_seg,
        #                           mode='TrainDataset_per_sq', prompt=args.prompt)  # TrainDataset_per_sq
        # vcod_test_dataset = VCOD_Test_EPFlow(args, args.data_path, transform=transform_test, transform_msk=transform_test_seg,
        #                          mode='TestDataset_per_sq', prompt=args.prompt)
        # vcod_test_dataset = VCOD(args, args.data_path, transform=transform_test, transform_msk=transform_test_seg,
        #                          mode='TestDataset_per_sq', prompt=args.prompt)

        nice_train_loader = DataLoader(vcod_train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(vcod_test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        '''end'''

    elif args.dataset == 'cod':
        '''cod data'''
        '''解决选点的位置要从原始图像大小去选择坐标的问题2024/8/21'''
        data_path = '/home/fabian/BRL/zhangxin/Datasets/COD'
        cod_train_dataset = COD(args, data_path, transform = transform_train, transform_msk= transform_train_seg, mode = 'TrainDataset', prompt=args.prompt)
        cod_test_dataset = COD(args, data_path, transform = transform_test, transform_msk= transform_test_seg, mode = 'TestDataset', prompt=args.prompt)

        nice_train_loader = DataLoader(cod_train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(cod_test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        '''end'''

    elif args.dataset == 'vcod_test':
        '''vcod_test data'''
        # vcod_test_dataset = VCOD(args, args.data_path, transform = transform_test, transform_msk= transform_test_seg, mode = test_data_name, prompt=args.prompt)
        '''解决选点的位置要从原始图像大小去选择坐标的问题2024/8/21'''
        vcod_test_dataset = VCOD_new(args, args.data_path, transform = transform_test, transform_msk= transform_test_seg, mode = test_data_name, prompt=args.prompt)

        nice_train_loader = None
        nice_test_loader = DataLoader(vcod_test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        '''end'''

    elif args.dataset == 'vos':
        '''解决选点的位置要从原始图像大小去选择坐标的问题2024/8/21'''
        vcod_test_dataset = VCOD_new(args, args.data_path, transform = transform_test, transform_msk= transform_test_seg, mode = 'DAVIS2016', prompt=args.prompt)

        nice_train_loader = None
        nice_test_loader = DataLoader(vcod_test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        '''end'''


    else:
        print("the dataset is not supported now!!!")
        
    return nice_train_loader, nice_test_loader