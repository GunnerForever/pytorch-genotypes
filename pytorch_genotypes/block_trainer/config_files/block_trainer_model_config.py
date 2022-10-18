def get_block_trainer_config():
    return {
        "lr": 5e-4,
        "batch_size": 256,
        "max_epochs": 200,
        "weight_decay": 1e-5,
        "add_batchnorm": True,
        "input_dropout_p": None,
        "enc_h_dropout_p": 0.1,
        "dec_h_dropout_p": 0.1,
        "val_proportion": 0.1,
        "model/activation": "LeakyReLU",
        "model/rep_size": 256,
        "model/enc_layers": [700, 400],
        "model/dec_layers": [400, 700],
    }
