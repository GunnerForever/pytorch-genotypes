def get_block_trainer_config():
    return {
        "lr": 1e-3,
        "batch_size": 256,
        "max_epochs": 300,
        "weight_decay": 1e-5,
        "add_batchnorm": True,
        "input_dropout_p": None,
        "enc_h_dropout_p": None,
        "dec_h_dropout_p": None,
        "model/activations": "LeakyReLU",
        "model/rep_size": 256,
        "model/enc_layers": [1000, 400],
        "model/dec_layers": [],
    }
