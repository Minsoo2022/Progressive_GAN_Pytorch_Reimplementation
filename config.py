def get_train_config(opt):
    config = {}
    config['max_iter_list'] = [20000, 20000, 20000, 50000, 50000, 10000, 100000, 100000, 100000]
    config['depth_list'] = [512,512,512,512,256,128,64,32,16]
    config['ch_Latent'] = 512
    config['batch_size_list'] = [256, 256, 128, 128, 128, 64, 32, 16, 8]
    if opt.debug:
        config['max_iter_list'] = [10, 10, 10, 10, 10, 10, 10, 10, 10]
        config['batch_size_list'] = [4, 4, 4, 4, 4, 4, 2, 2, 2]
    return config