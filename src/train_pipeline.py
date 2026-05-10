import huggingface_hub
import sacremoses
import os
import numpy as np
import train_code as tc
import datasets

DATASET_NAME = "disi-unibo-nlp/Pile-NER-biomed-IOB"

hyperparam = {
    'RANDOM_SEED': 42,
    'VALIDATION_SIZE': 0.02,
    'TEST_SIZE': 0.05,
    'MAX_ENTITIES_PER_SAMPLE': 6
}
np.random.seed(hyperparam['RANDOM_SEED'])


def prepare_dataset():
    # Log into HF
    token = os.environ["HF_TOKEN"]
    huggingface_hub.login(token)

    # Initialize detokenizer
    detok = sacremoses.MosesDetokenizer(lang="en")

    # Prepare Pile dataset
    pile_ds = tc.create_sft_ds(
        ds=datasets.load_dataset(DATASET_NAME),
        max_entities=hyperparam['MAX_ENTITIES_PER_SAMPLE'],
        detok=detok,
        hyperparam=hyperparam
    )
    pile_ds = tc.get_split_ds(pile_ds, hyperparam['VALIDATION_SIZE'], hyperparam['TEST_SIZE'], hyperparam['RANDOM_SEED'])
    
    return pile_ds

if __name__ == "__main__":
    pile_ds = prepare_dataset()

    print('PILE DATASET:')
    print(pile_ds)
    print('\nRANDOM SAMPLE:')
    print(pile_ds['test'][5])