def get_train_config(opt):
    config = {}
    if opt.train_type == 0 :
        config['max_iter_list'] = [20000, 20000, 20000, 50000, 50000, 100000, 100000, 100000, 100000]
        config['batch_size_list'] = [256, 256, 128, 128, 128, 64, 32, 16, 8]
    elif opt.train_type == 1 :
        config['max_iter_list'] = [20000, 20000, 20000, 20000, 50000, 50000, 50000, 50000, 50000]
        config['batch_size_list'] = [256, 128, 64, 64, 64, 32, 16, 8, 4]
    config['depth_list'] = [512,512,512,512,256,128,64,32,16]
    config['ch_Latent'] = 512
    if opt.debug:
        config['max_iter_list'] = [10, 10, 10, 10, 10, 10, 10, 10, 10]
        config['batch_size_list'] = [4, 4, 4, 4, 4, 4, 2, 2, 2]
    return config