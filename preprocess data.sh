#!/usr/bin/env bash

pwd
preprocessed=1

export data="./data"
export root="./glove_experiment"
export glove_dir="./glove"

#mkdir -p $root
#mkdir -p $data

if [[ preprocessed -ne 1 ]];
then
  echo "Starting Preprocessing..."
  python ././../OpenNMT-py/preprocess.py \
      -train_src $data/train.src.txt \
      -train_tgt $data/train.src.txt \
      -valid_src $data/valid.src.txt \
      -valid_tgt $data/valid.src.txt \
      -save_data $root/data

  echo "Finished Preprocessing!"
fi


././../OpenNMT-py/tools/embeddings_to_torch.py -emb_file "$glove_dir/glove.6B.100d.txt" \
                               -dict_file "$root/data.vocab.pt" \
                               -output_file "$root/embeddings"


