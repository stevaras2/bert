# Link for pre-trained BERT model
https://drive.google.com/file/d/17aac1zyF8ez7rTqYCUJEVZPslULjtLa6/view?usp=sharing

# Links for sentence embeddings extracted from different layers of the fine-tuned BERT(fine-tune using MRPC dataset).

Layer -1(final hidden layer): https://drive.google.com/open?id=1ZB9FOwD1n25uAByxnOXXYrHTy6YIdW1I <br /> 
Layer -2: https://drive.google.com/open?id=1uzNN62VBeoT0BQxMR34I94dtYzrdHA0_ <br /> 
Layer -3: https://drive.google.com/open?id=17zD9KJA5MsWOl5JfY9n8nkai3Y8mgQZd <br /> 
Layer -4: https://drive.google.com/open?id=1CPomvoNJd2eJcFD9iBy5N5j4cBH7KE_l <br /> 



# Links for sentence embeddings extracted from different layers of the fine-tuned BERT(fine-tune using our dataset).

Layer -1(final hidden layer): https://drive.google.com/open?id=1byaD-KN2gj-cZ1w9NrervlaJNFPgf1mC <br /> 
Layer -2: https://drive.google.com/open?id=1J7_gFa3vhM6__a-OPfWSMgST5lVntyFb <br /> 
Layer -3: https://drive.google.com/open?id=1SbUgVZniwA4618uce97yxzv8BLg9kAaq <br /> 
Layer -4: https://drive.google.com/open?id=1-unAkcIvBYsSMtTo9VkJEhSkh0Sejglk <br /> 

# Link for the fine-tuned model in our dataset

https://drive.google.com/open?id=1ka-3CFXt5bUbkEqAgzYVomGTPern2XTv

# Command to reproduce results

In order to reproduce the results, you should create the my_dataset_output folder to save the fine-tuned model. The command to fine tune the model is the following:<br>

python run_classifier.py <br>--task_name=MRPC <br>--do_train=true <br>--do_eval=true <br>--data_dir=path\to\dataset <br> --vocab_file=path\to\vocab.txt <br>--bert_config_file=path\to\bert_config.json <br>--init_checkpoint=path\to\bert_model.ckpt <br>--max_seq_length=128 <br>--train_batch_size=8 <br>--learning_rate=2e-5 <br>--num_train_epochs=3.0 <br>--output_dir=my_dataset_output

<br><br>
In order to extract the embeddings of the fine-tuned model you should type the following command:<br>

python extract_features.py<br> --input_file=path\to\sentences_list.txt <br>--output_file=output.json <br>--vocab_file=path\to\vocab.txt <br>--bert_config_file=path\to\bert_config.json <br>--init_checkpoint=path\to\bert_model.ckpt  <br>--layers=-1 or -2 or etc  <br>--max_seq_length=128 <br>--batch_size=8

# Preproccess dataset before fine-tune

The create_dataset_for_fine_tune.py is the script that preproccess the dataset before the fine-tuning. The command to execute this script(in Windows) is <br>
python create_dataset_for_fine_tune.py -ts train_sentences1.csv -tp train_papers

# Train classifier/summarizer

The script train_text_summarizer.py train the summarizer that will be used. In order to execute this script in Windows, type the following command:<br>
python train_text_summarizer.py -s path\to\sentences_list.txt -o path\to\embedding\file -ts path\to\train_set -sp path\to\save\summarizer


# Evaluation summarization approaches

The script summarization.py produces the summaries of the papers in the test set and evaluate them. To execute this script on Windows type this command:<br>
python summarization.py -s path/to/test_sentences_list.txt -o path/to/json_file_where_embeddings_will_be_saved -v path/to/vocab.txt -c path/to/bert_config.json -m path/to/fine-tune/model

# Results files

The results could be found in the tsv files of this repository.
