import os

import fire

from lxuechen_utils import utils


def main(
    copy_init="/Users/xuechenli/Desktop/dump/tests/prefix-tuning",
    past_init="/Users/xuechenli/Desktop/dump/tests/prefix-tuning2",

    quant="tok_logprobs",
):
    assert quant in ("tok_logprobs", "lin_logprobs")

    record_path = os.path.join(copy_init, 'log_history.json')
    record = utils.jload(record_path)

    plots = []

    # copy_init
    xtrain = [this_dict["step"] for this_dict in record]
    ytrain = [this_dict["train"]["model"][quant] for this_dict in record]
    plots.append({'x': xtrain, 'y': ytrain, 'label': 'train (copy)'})
    del xtrain, ytrain

    xval = [this_dict["step"] for this_dict in record]
    yval = [this_dict["val"]["model"][quant] for this_dict in record]
    plots.append({'x': xval, 'y': yval, 'label': 'val (copy)'})
    del xval, yval

    xtest = [this_dict["step"] for this_dict in record]
    ytest = [this_dict["eval"]["model"][quant] for this_dict in record]
    plots.append({'x': xtest, 'y': ytest, 'label': 'eval (copy)'})
    del xtest, ytest

    del record
    record_path = os.path.join(past_init, 'log_history.json')
    record = utils.jload(record_path)

    # past_init
    xtrain = [this_dict["step"] for this_dict in record]
    ytrain = [this_dict["train"]["model"][quant] for this_dict in record]
    plots.append({'x': xtrain, 'y': ytrain, 'label': 'train (past)', 'linestyle': '-.'})
    del xtrain, ytrain

    xval = [this_dict["step"] for this_dict in record]
    yval = [this_dict["val"]["model"][quant] for this_dict in record]
    plots.append({'x': xval, 'y': yval, 'label': 'val (past)', 'linestyle': '-.'})
    del xval, yval

    xtest = [this_dict["step"] for this_dict in record]
    ytest = [this_dict["eval"]["model"][quant] for this_dict in record]
    plots.append({'x': xtest, 'y': ytest, 'label': 'eval (past)', 'linestyle': '-.'})
    del xtest, ytest

    img_path = os.path.join('.', 'gpt2stuff', 'plots', 'untie_init', f'{quant}')
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    utils.plot(
        img_path=img_path,
        plots=plots,
        options={'ylabel': quant, 'xlabel': 'steps'}
    )


if __name__ == "__main__":
    # python -m gpt2stuff.plots.untie_init
    fire.Fire(main)
