from data.custom_datasets import BSMSegLoader
from torch.utils.data import DataLoader

data_dict = {
    'BSM': BSMSegLoader
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    batch_size = args.batch_size


    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            pin_memory=True)
        return data_set, data_loader
