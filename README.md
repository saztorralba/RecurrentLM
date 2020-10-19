# recurrent-lm
## Description
Code and scripts for training, testing and sampling auto-regressive recurrent language models on PyTorch with RNN, GRU and LSTM layers.

## Caveat emptor
This repository is for educational and showcasing purposes only. The contents are provided 'as is' without any implied warranty, and disclaiming liability for damages resulting from using it.

## Functionalities
The code provides 3 functionalities: training an LM, evaluating an LM and sampling sequences from an LM. Both word-level and character-level LMs can be trained. Training can be performed in CPU or GPU, testing and sampling is performed on CPU.

### Training
Training is done with the ``train_lm.py`` script. Run ``python train_lm.py --help`` for details on the existing arguments. An example run is provided here:

```
python train_lm.py --input_file data/librispeech-train-sentences.list --output_file output/final_model.pytorch --vocabulary data/librispeech-train-top10k.voc --cv_percentage 0.1 --epochs 10 --batch_size 8 --embedding_size 512 --hidden_size 512 --num_layers 1 --learning_rate 0.8 --ltype lstm --max_length 50
```

It allows for LSTM, GRU, and RNN models by setting ``--ltype`` to lstm, gru, rnn and dnn respectively. All networks can be trained using truncated backpropagation through time. If no vocabulary is provided, a character-level LM will be trained.

### Testing
Testing is done with the ``test_lm.py`` scripts. Run ``python test_lm.py --help`` for details on the existing argument. An example run is provided here:

```
python test_lm.py --input_file data/librispeech-test-sentences.list --model_file output/final_model.pytorch --batch_size 128 --verbose 0
```

It will report the overall log-probability and perplexity (PPL) for the whole set, ``--verbose`` can be set to 1 or 2 to report results per sentence or word respectively.

### Sampling
Generating random samples from a model is done with ``sample_lm.py`` script. Run ``python sample_lm.py --help`` for details on the existing arguments. An example run is provided here:

```
python sample_lm.py --model_file output/final_model.pytorch --num_sequences 10 --topk 10000 --distribution exponential
```

It will generate as many sentences as given by ``--num_sequences``, and will report their log-probabilities and perplexities. The ``--verbose`` argument can be used as in testing. The argument ``--start_with`` can be used to generate sentences starting with some words provided.

## Dependencies

The code has been tested in the following environment, with the main following dependencies and versions:

* python --> 3.6.7 & 3.8.3
* torch --> 1.4.0
* numpy --> 1.19.2

## Evaluation

### Data

A train and test subsets from Librispeech is provided, along with a selection of the 10,000 most common words as vocabulary. This data has been extracted from the transcripts of the Librispeech data provided in [OpenSLR](http://www.openslr.org/12/).

### Results

These are some results achieved on this train and test subsets using different recurrent layers, compared to 2-gram and 5-gram trained with SRILM.

Method  | Perplexity
--------|----------
2-gram  | 252.49 
5-gram  | 191.60
RNN LM  | 159.43 
GRU LM  | 133.42 
LSTM LM | 132.19 

### Generating samples

These are examples of random sentences generated using the autoregressive LSTM LM model with an exponential sampler:

> I SHOULD HAVE DONE THAT IT WAS EVEN THE VOICE OF THE COURTESY OF THE MAN HE WOULD ALWAYS TELL ME HOW I HAVE BEEN PAY FOR THE LETTER

> WHEN THE OTHERS CAME OUT OF THE HAMMER HE HEARD THE DAWN OF THE STORY OF THE PALACE

> AND HIS WIFE HAS BEEN A GREAT DEAL BETTER THAN HER MOTHER NOW SHE IS THE BEST OF ALL HIS HEART AND DEATH

> AND TELL ME SAID I TO BE THERE IN THE HOUSE OF A MAN WHO IS A WOMAN OF YOUR CHARACTER

> IT IS THE NEAREST POINT OF POLITICS TO ITS OWN TASKS

> THAT THE DUST WHICH WAS TO HOLD UP THE HILL

> THE TALL KING FELT HER HIGH AND HAPPY MOTHER AND HER MOTHER WAS SO UNHAPPY THAT HE MAY COME TO THE WORLD

> BUT THE FOOTSTEPS WERE TAKEN AND THEN THEY WENT ON THE SALMON TOOK THE GOLDEN BALL INTO THE SEA

> FOR THE LAST TIME HE SAW THE VERY CREW OF THE WHITE HOUSE

> I SUPPOSE YOU WERE TO GO ON ASKED THE CORONER

> I WILL BESTOW A PART OF A CABINET NO MATTER IF IT BE DONE TO TRIM THE STARS IN ORDER TO USE IT AS THE WEATHER COMBINATION

> IN THE MOST ARDENT MANNER THE WORD WAS THE LETTER OF THE PRINCESS OF THE HOUSE THE TWO CHILDREN WERE AT THE ENTRANCE OF THE TABLE

> IF YOU SHOULD HAVE TO MAKE THE BEST OF YOUR GOOD FORTUNE AND I WILL TELL YOU WHAT I HAVE TO FEAR

> AND IT IS NECESSARY TO SAY THAT THE ORDINARY LAWS OF THE UNIVERSE ARE PRODUCED BY THE INTELLECT

> AND HOW HE HAD NEVER BEEN UNDERSTOOD TO UTTER A PORTION OF HIS MIND THAT HE WAS LESS THAN ANY OF HIS MOST AMIABLE MEN

> BUT AS SHE SAT WAITING FOR THE SLEDGE FOR A MOMENT WHO STOOD IN A CORNER ROOM THE WOODEN PRINCESS TOOK HER UP AND THE OTHERS COULD HEAR THE THREE MOMENTS

> THE WIND WAS STILL AND LOW AND THE RIGID IRON RAYS ROLLED ON THE FLOOR GREW SO BEAUTIFUL THAT THE WIND COULD NOT BE OBSCURE

> THEY WERE SO EXCITED THAT THEY WERE ALMOST TIRED OF THE SAME CHARACTER

> AND A VISIT TO A PLACE IN A WEEK OF FIFTEEN MILES A BOARD OF LARGER AND HIGH A COUNTIES

