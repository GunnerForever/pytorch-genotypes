def get_block_trainer_config():
    return {
        "lr": 5e-5,
        "batch_size": 256,
        "max_epochs": 200,
        "weight_decay": 1e-5,
        "add_batchnorm": True,
        "input_dropout_p": None,
        "enc_h_dropout_p": 0,
        "dec_h_dropout_p": 0,
        "val_proportion": 0.1,
        "use_standardized_genotype": False,
        "model/activation": "LeakyReLU",
        "model/rep_size": 32,
        "model/enc_layers": [2000, 500],
        "model/dec_layers": [500, 200],
    }
