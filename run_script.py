from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Flow type strings for convenience
FLOW_TYPE_IDENTITY_FLOW = "IDENTITY_FLOW"
FLOW_TYPE_IAF = "IAF"

# Encoder type strings for convenience
NONCONV_ENCODER = "NONCONV_ENCODER"
IAF_ENCODER = "IAF_ENCODER"
SNF_ENCODER = "SNF_ENCODER"

# Encoder type strings for convenience
NONCONV_DECODER = "NONCONV_DECODER"
IAF_DECODER = "IAF_DECODER"
SNF_DECODER = "SNF_DECODER"


def build_parameter_dict():
    parameter_dictionary = {
        "name": "vae_iaf",
        "restart_filename": None,
        "num_epochs": 2500,
        "model_params": {
            "warmup_epochs": 200.0,
            "approx_post_offset": -3.0,
            "num_latent_units": 64,
            "num_is_samples_nll": 1000,
            "encoder_params": {
                "encoder_type": IAF_ENCODER,
                # "nn_hidden_layers": 3 * [1000],
            },
            "flow_params": {
                "flow_type": FLOW_TYPE_IAF,
                "flow_layers": 8,
                "flow_shift_only": False,
                "cmade_hidden_layers": [1920, 1920],
                "cmade_batchnorm": True,
                "cmade_context_SNF_like": True,
                "context_size": 1920
            },
            "decoder_params": {
                "decoder_type": IAF_DECODER,
                # "nn_hidden_layers": 3 * [1000],
            },
            "opt_params": {
                "base_learning_rate": 0.001,
                "learning_rate_decay_epochs": 800,
                "decay_scale_factor": 0.5,
            },
            "hais_params": None,
            # "hais_params": {
            #     "anneal_steps": 2000,
            #     "leapfrog_steps": 10,
            #     "leapfrog_stepsize": 0.1,
            #     "num_ais_chains": 8,
            #     "use_encoder": False,
            #     "target_acceptance_rate": 0.65,
            #     "avg_acceptance_slowness": 0.8,
            #     "stepsize_min": 0.0001,
            #     "stepsize_max": 0.9,
            #     "stepsize_inc": 1.02,
            #     "stepsize_dec": 0.98
            # },
        },
        "data_params": {
            "batchsize_train": 200,
            "batchsize_val": 0,
            "batchsize_test": 200,
            "trainsize": 60000,
            "valsize": 0,
            "testsize": 10000,
        },
    }
    return parameter_dictionary


def run_experiment():
    from code.manager import Manager
    manager = Manager(**build_parameter_dict())
    manager.run_model()


if __name__ == '__main__':
    run_experiment()
