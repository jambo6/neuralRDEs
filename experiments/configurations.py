"""
configurations.py
===========================
Contains the hyper-parameter and final run configurations for each dataset.
"""

default = {
    'test': {
        'model_type': ['nrde'],
        'depth': [1],
        'step': [1000],
        'num_layers': [2],
        'hidden_dim': [1],
        'hidden_hidden_multiplier': [1],
        'num_layers': [1],
        'seed': [0],
    },
    'hyperopt': {
        'model_type': ['nrde'],
        'depth': [1],
        'step': [1],
        'hidden_dim': [16, 32, 64],
        'hidden_hidden_multiplier': [2, 3, 4],
        'num_layers': [1, 2, 3],
        'seed': [1234],
    },
    'main': {
        # Data
        'data__batch_size': [128],
        # Main
        'model_type': ['nrde'],
        'depth': [1, 2, 3],
        'step': [1, 2, 3, 5, 10, 20, 50],
        'hyperopt_metric': ['acc'],
        'seed': [111, 222, 333],
    },
    'bidmcmain': {
        'model_type': ['nrde'],
        'data__batch_size': [512],
        'hyperopt_metric': ['loss'],
        'depth': [1, 2, 3],
        'step': [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
        'seed': [111, 222, 333],
    },
}


configs = {
    # UEA
    'UEA': {
        'EigenWorms': {
            'test': {
                **default['test']
            },
            'hyperopt': {
                **default['hyperopt'],
                'data__batch_size': [1024],
                'step': [36]    # Total steps [500]
            },
            'main': {
                **default['main'],
                'data__batch_size': [1024],
                'data__adjoint': [True],
                'hyperopt_metric': ['acc'],
                'depth': [1, 2, 3],
                'step': [1, 2, 4, 6, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
            },
        },
    },

    # TSR
    'TSR': {
        'BIDMC32SpO2': {
            'hyperopt': {
                **default['hyperopt'],
                'data__batch_size': [512],
                'step': [8],
            },
            'main': {
                **default['bidmcmain'],
                'data__adjoint': [True]
            }
        },

        'BIDMC32RR': {
            'hyperopt': {
                **default['hyperopt'],
                'data__batch_size': [512],
                'step': [8],
            },
            'main': {
                **default['bidmcmain'],
                'data__adjoint': [True]
            }
        },

        'BIDMC32HR': {
            'hyperopt': {
                **default['hyperopt'],
                'data__batch_size': [512],
                'step': [8],
            },
            'main': {
                **default['bidmcmain'],
                'data__adjoint': [True]
            }
        },
    },

}

