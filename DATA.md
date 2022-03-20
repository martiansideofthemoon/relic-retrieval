# Dataset Description

RELiC consists of three `.json` files: `train.json`, `val.json`, and `test.json`. Each of these files has the following format:

```json
{
   "book_title": {
      "quotes": {
         "unique_id1": [
            [
               "pre_sent1",
               "pre_sent2",
               "pre_sent3",
               "pre_sent4"
            ],
            1,
            4,
            [
               "sub_sent1",
               "sub_sent2",
               "sub_sent3",
               "sub_sent4"
            ]
         ]
      },
      "sentences": [
         "sent1",
         "sent2",
         "sent3"
      ],
      "candidates": {
         "1_sentence": [
            0,
            1,
            2,
            3,
            4,
            5,
            6
         ],
         "2_sentence": [
            0,
            1,
            2,
            3,
            4,
            5
         ],
         "3_sentence": [
            0,
            1,
            2,
            3,
            4
         ],
         "4_sentence": [
            0,
            1,
            2,
            3
         ],
         "5_sentence": [
            0,
            1,
            2
         ]
      }
   }
}

```
where `dataset["book_title"]` is a dictionary with 3 keys: `"quotes"`, `"sentences",` and `"candidates"`.

`dataset["book_title"]["quotes"]` is a dictionary where each key is a unique identifier for one instance of the dataset. `dataset["book_title"]["quotes"]["unique_id"]` is a list with 4 items:

0. A list of the sentences preceding the literary quotation
1. The index in `dataset["book_title"]["sentences"]` of the literary quotation
2. The length of the literary quotation in number of sentences
3. A list of the sentences after the literary quotation

`dataset["book_title"]["sentences"]` is a list of sentences in `"book_title"`

`dataset["book_title"]["candidates"]` is a dictionary with 5 keys, each corresponding to a list of valid starting indices for literary quotations of length n (in sentences). For example, `dataset["book_title"]["candidates"]["3_sentence"]` is a list of valid starting indices for literary quotations of 3 sentences.

To reconstruct the list of sentences of the literary quotation in `dataset["book_title"]["quotes"]["unique_id"]`, you would do the following:
``` python
start_idx = dataset["book_title"]["quotes"]["unique_id"][1]
end_idx = dataset["book_title"]["quotes"]["unique_id"][1] + dataset["book_title"]["quotes"]["unique_id"][2]
quote_sents = dataset["book_title"]["sentences"][start_idx:end_idex]
```
