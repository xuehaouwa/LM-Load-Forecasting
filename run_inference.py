from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import BlenderbotSmallForConditionalGeneration, AutoTokenizer
from transformers import BlenderbotForConditionalGeneration
from transformers import T5ForConditionalGeneration
from transformers import PegasusForConditionalGeneration, ProphetNetForConditionalGeneration
from transformers import LEDForConditionalGeneration, PegasusTokenizer, BigBirdPegasusForConditionalGeneration
import argparse
import os
import torch
import jsonlines
from metric import metric_with_missing_rate
from tqdm import tqdm


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('-t', '--test_file',
                      default='',
                      type=str)
    args.add_argument('-m', '--model_path', default='results',
                      type=str)
    args.add_argument('--model_name', default='bart',
                      type=str)
    args.add_argument('-s', '--save_path', default='results',
                      type=str)
    args.add_argument('--building',
                      type=str)
    args.add_argument('-p', '--prediction_length', default=4,
                      type=int)
    return args.parse_args()


def get_tokens(input_file, tokenizer, max_length):
    token_ids = []

    for sent in input_file:
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=max_length,  # Pad & truncate all sentences.
            padding='max_length',
            return_attention_mask=False,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        # Add the encoded sentence to the list.
        token_ids.append(encoded_dict['input_ids'])
    # #
    token_ids = torch.cat(token_ids, dim=0)

    return token_ids


def batchpredict(batch_example, tokenizer):
    # inputs = [example]
    model_inputs = tokenizer(batch_example, max_length=1024, truncation=True, padding=True)

    beam_outputs = model.generate(
        input_ids=torch.LongTensor(model_inputs['input_ids']).cuda(),
        attention_mask=torch.LongTensor(model_inputs['attention_mask']).cuda(),
        max_length=64,
        early_stopping=True,
        num_beams=None,
        num_return_sequences=1,
        no_repeat_ngram_size=2
    )
    sent = tokenizer.batch_decode(beam_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print(sent)

    return sent


def predict(example, tokenizer):
    inputs = [example]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding=True)

    beam_outputs = model.generate(
        input_ids=torch.LongTensor(model_inputs['input_ids']).cuda(),
        attention_mask=torch.LongTensor(model_inputs['attention_mask']).cuda(),
        max_length=64,
        early_stopping=True,
        num_beams=4,
        num_return_sequences=1,
        no_repeat_ngram_size=2
    )
    sent = tokenizer.batch_decode(beam_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    return sent


def clean(x):
    x = x.replace("\n", "")
    return x


def recurrent_predict(input_sentences, tokenizer, horizon=4):
    output = []
    for i in range(horizon):
        example = ' '.join(input_sentences)
        predicted = predict(example, tokenizer)
        input_sentences.append(predicted[0])
        input_sentences = input_sentences[1:]
        output.append(predicted[0])

    return output


def batch_recurrent_predict(input_sentences, tokenizer, horizon=4):
    batch_size = len(input_sentences)
    for i in range(horizon):
        example = []
        for b in range(batch_size):
            example.append(' '.join(input_sentences[b]))
        predicted = batchpredict(example, tokenizer)
        for b in range(batch_size):
            input_sentences[b].append(predicted[b])
            input_sentences[b] = input_sentences[b][1:]


if __name__ == "__main__":

    args = get_args()
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    setting = '{}_{}_predict{}'.format(
        args.model_name,
        args.building,
        args.prediction_length)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model_name == 'bart':
        model = BartForConditionalGeneration.from_pretrained(args.model_path)
        tokenizer = BartTokenizer.from_pretrained(args.model_path)
    elif args.model_name == 'blendersmall':
        model = BlenderbotSmallForConditionalGeneration.from_pretrained(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    elif args.model_name == 'blender':
        model = BlenderbotForConditionalGeneration.from_pretrained(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    elif args.model_name == 'pegasus':
        model = PegasusForConditionalGeneration.from_pretrained(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    elif args.model_name == 'prophet':
        model = ProphetNetForConditionalGeneration.from_pretrained(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    elif args.model_name == 'led':
        model = LEDForConditionalGeneration.from_pretrained(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    elif args.model_name == 'T5':
        model = T5ForConditionalGeneration.from_pretrained(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    elif args.model_name == 'bigbird':
        model = BigBirdPegasusForConditionalGeneration.from_pretrained(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model.to(device)
    #
    gt_lines = []
    predicted_lines = []
    with jsonlines.open(args.test_file) as reader:
        for obj in tqdm(reader):
            predicted = recurrent_predict(obj['input'], tokenizer, horizon=args.prediction_length)
            gt_lines.extend(obj['target'])
            predicted_lines.extend(predicted)

    rmse, mae, missing_rate = metric_with_missing_rate(gt_lines, predicted_lines)
    print(f"{rmse=}")
    print(f"{mae=}")
    print(f"{missing_rate=}")
    f = open("result_forecast.txt", 'a')
    f.write(setting + "  \n")
    f.write('mae:{}, rmse:{}, missing_rate:{}'.format(rmse, mae, missing_rate))
    f.write('\n')
    f.write('\n')
    f.close()
    with open(os.path.join(args.save_path, "predicted.txt"), "w") as f:
        f.writelines([x + "\n" for x in predicted_lines])
        f.close()

