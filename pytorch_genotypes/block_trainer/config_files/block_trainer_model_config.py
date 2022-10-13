def get_block_trainer_config():
    return {
        "lr": 1e-3,
        "batch_size": 256,
        "max_epochs": 10,
        "weight_decay": 1e-5,
        "add_batchnorm": True,
        "input_dropout_p": None,
        "enc_h_dropout_p": 0.1,
        "dec_h_dropout_p": 0.1,
        "val_proportion": 0.1,
        "model/activations": "LeakyReLU",
        "model/rep_size": 256,
        "model/enc_layers": [1000, 400],
        "model/dec_layers": [400, 1000],
    }
