def get_train_config():
    config = {}
    config['max_iter_list'] = [50000,50000,50000,50000,50000,50000,50000,50000,50000]
    config['depth_list'] = [512,512,512,512,256,128,64,32,16]
    config['ch_Latent'] = 512
    config['batch_size_list'] = [64, 64, 64, 64, 64, 32, 16, 8, 4]
    return config