def get_block_trainer_config():
    return {
        "lr": 0.00008712,
        "batch_size": 256,
        "max_epochs": 200,
        "weight_decay": 0.000142,
        "add_batchnorm": True,
        "input_dropout_p": 0.2,
        "enc_h_dropout_p": 0.2,
        "dec_h_dropout_p": 0,
        "val_proportion": 0.1,
        "use_standardized_genotype": False,
        "model/activation": "GELU",
        "model/rep_size": 128,
        "model/enc_layers": [1300, 1100],
        "model/dec_layers": [1100, 1300],
    }
