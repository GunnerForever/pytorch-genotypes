def get_block_trainer_config():
    return {
        "lr": 1e-3,
        "batch_size": 256,
        "max_epochs": 1000,
        "weight_decay": 1e-5,
        "add_batchnorm": True,
        "input_dropout_p": 0.2,
        "enc_h_dropout_p": 0.2,
        "dec_h_dropout_p": 0.2,
        "val_proportion": 0.1,
        "use_standardized_genotype": False,
        "model/activation": "GELU",
        "model/rep_size": 128,
        "model/enc_layers": [500, 200],
        "model/dec_layers": [200, 200],
    }
