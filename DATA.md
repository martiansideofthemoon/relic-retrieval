# Dataset Description

RELiC consists of three `.json` files, `train.json`, `val.json`, and `test.json`. Each of these files has the following format:

```json
"book_title": {"quotes": {
                          "unique_id1": [
                                          ["pre_sent1", "pre_sent2", "pre_sent3", "pre_sent4"],
                                          quote_idx,
                                          quote_len,
                                          ["sub_sent1", "sub_sent2", "sub_sent3", "sub_sent4"]
                                        ]
                         },
              "sentences": ["sent1", "sent2", "sent3", ...],
              "candidates": {"1_sentence": [0,1,2,...],
                            "2_sentence": [0,2,4,...],
                            "3_sentence": [0,3,6,...],
                            "4_sentence": [0,4,8,...],
                            "5_sentence": [0,5,10,...]
                            }
              }

```
