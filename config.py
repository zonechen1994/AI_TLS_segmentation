import argparse
model_name = 'TLSNet_efficientNet'
def getConfig():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=100, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or Adam')
    parser.add_argument('--augmentation',
                        default=False, help='choose to do random flip rotation')
    parser.add_argument('--batchsize', type=int,
                        default=64, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--arch', type=str,
                        default='0', help='backbone network')
    parser.add_argument('--cancer_type', type=str,
                        default='ESCC', help='cancer_type')
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,
                        default=10, help='every n epochs decay learning rate')
    parser.add_argument('--train_path', type=str,
                        default='./TLS_segmentation/dataset/TrainDataset',
                        help='path to train TLS segmentation model')
    parser.add_argument('--test_path', type=str,
                        default='./TLS_segmentation/dataset/TestDataset/',
                        help='path to test TLS segmentation model')
    parser.add_argument('--train_save', type=str,
                        default='./model_pth/'+model_name+'/')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--stage', type=str, default='val')
    parser.add_argument('--refine_channels', type=int, default=16)
    parser.add_argument('--model', type=str, default='efficientnet-b0')
    parser.add_argument('--start_dir', type=int, default=0)
    parser.add_argument('--end_dir', type=int, default=0)
    opt = parser.parse_args()
    return opt

