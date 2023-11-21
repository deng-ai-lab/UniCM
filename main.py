from run_train import run_training_image, run_training_im2im
from config.cfg_image import ConfigIm2Im2, ConfigImage, ConfigImage2, ConfigImage3
from run_test import run_testing_image,run_testing_im2im
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='DCM', choices=['DCM','DCM-MC','CCM','CCM-OT','PCM'], help='method name')
    parser.add_argument('--data', type=str, default='cifar10', choices=['cifar10','Celeba','AFHQ'], help='data name')
    parser.add_argument('--task', type=str, default='uncondition', choices=['uncondition', 'Im2Im'], help='task name')
    parser.add_argument('--train', type=bool, default=True)
    args = parser.parse_args()

    if args.model == 'DCM' or args.model == 'DCM-MS':

        cfg = ConfigImage()
    
    elif args.model == 'CCM' or args.model == 'CCM-OT':

        if args.task == 'uncondition':
            
            cfg = ConfigImage2()
        else:
            cfg = ConfigIm2Im2()
    
    elif args.model == 'PCM':
            
            cfg = ConfigImage3()

    else:
         raise ValueError('{args.model} has not been implemented yet.')
    
    cfg.cm_type = args.model
    cfg.version = args.data


    if args.train == True:
         
        if args.task == 'uncondition':
            run_training_image(cfg)
        else:
            run_training_im2im(cfg)
    else:
        if args.task == 'uncondition':
            run_testing_image(cfg)
        else:
            run_testing_im2im(cfg)
