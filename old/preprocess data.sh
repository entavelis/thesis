#!/usr/bin/env bash

pwd
removeEverything=0
tutorial_data=0
preprocessed=1
emb_created=0
shouldItrain=1

if [[ removeEverything -eq 1 ]];
then
    echo "Removing files..."
    rm ./glove_experiment/*
fi


export data="./data"
export root="./glove_experiment"
export glove_dir="./glove"

#mkdir -p $root
#mkdir -p $data

train_src="$data/train.src.txt"
train_tgt="$data/train.src.txt"
valid_src="$data/valid.src.txt"
valid_tgt="$data/valid.src.txt"


if [[ tutorial_data -eq 1 ]];
then
  echo "Using OpenNMT tutorial data"

  train_src="$data/src-train.txt"
  train_tgt="$data/tgt-train.txt"
  valid_src="$data/src-val.txt"
  valid_tgt="$data/tgt-val.txt"

fi


if [[ preprocessed -ne 1 ]];
then
  # Added shared_vocab

  echo "Starting Preprocessing..."
  python ././../OpenNMT-py/preprocess.py \
      -train_src $train_src \
      -train_tgt $train_src \
      -valid_src $valid_src \
      -valid_tgt $valid_tgt \
      -share_vocab \
      -save_data $root/data
#     -src_vocab_size 1000 \
#     -tgt_vocab_size 1000
  echo "Finished Preprocessing!"

  python ./load_vocab.py
fi

if [[ emb_created -ne 1 ]];
then
  echo "Creating Embeddings..."
  python ./OpenNMT/tools/embeddings_to_torch.py -emb_file "$glove_dir/glove.6B.100d.txt" \
                                 -dict_file "$root/data.vocab.pt" \
                                 -output_file "$root/embeddings"

  echo "Embeddings created!"
fi

if [[ shouldItrain -eq 1 ]];
then
  echo "Starting training..."


  python ././../OpenNMT-py/train.py -save_model data/model \
  -data $root/data
  -batch_size 64 \
  -layers 2 \
  -rnn_size 200 \
  -input-feed 0 \
  -encoder-type "brnn" \
  -rnn-type "GRU" \
  -share_embeddings \
  -share_decoder_embeddings \
  -word_vec_size 100 \
  -pre_word_vecs_enc "$root/embeddings.enc.pt" \
  -pre_word_vecs_dec "$root/embeddings.dec.pt"

echo "Training done!"
fi


