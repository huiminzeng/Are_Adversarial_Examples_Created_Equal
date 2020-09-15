def exp_name(args):
    # set up comet experiment name
    # set model name first
    # model_type for cifar10
    if args.size == 18:
        model_type = 'resnet18'
    
    # mode_type for mnist
    elif args.size == 'SmallCNN':
        model_type = 'smallcnn'

    if args.mode == "baseline":
        experiment_name = model_type + "_" + args.mode + '_lr_' + str(args.lr) + '_lambda_' + str(args.beta) + '_seed_' + str(args.seed)
        print("training mode: ", experiment_name)
                
    elif args.mode == "margin":
        experiment_name = model_type + "_" + args.mode + '_lr_' + str(args.lr) + '_lambda_' + str(args.beta) + '_alpha_' + str(args.alpha) + '_seed_' + str(args.seed)
        print("training mode: ", experiment_name)
        
    return experiment_name