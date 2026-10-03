# Possibly incorrect labels in the CUAD test split

`test_split_questionable_labels.csv` lists 141 CUAD answers in 47 **test-split** contracts whose
text does not appear to meet the definition of the category it is labelled with.

## How the list was made

1. An automated contract-clause extractor labelled the test contracts.
2. Every CUAD answer that the extractor did not label with the same category was read by a
   reviewer against the category's definition in `category_descriptions.csv` (the written
   definition only - not the extractor's own guidance).
3. An answer is listed only when it plainly does not meet the definition. Answers that were cut
   off, redacted, or borderline were left out.
4. Each row is CUAD's own annotation: `answer_text` and `answer_start` are copied unchanged from
   the test data, so a row identifies exactly one existing label.

## Columns

| Column | Meaning |
|---|---|
| `contract_title` | The contract's title as in the CUAD test data |
| `category` | The CUAD category the answer is labelled with |
| `answer_start` | The answer's character offset, as in CUAD |
| `answer_text` | The answer's text, as in CUAD |
| `why it does not meet the CUAD category definition` | The reviewer's one-line reason |

## Caveats

- One reviewer's reading against the written definitions, not a legal review.
- Minimum Commitment accounts for 48 rows. Its definition asks for an amount one party must
  *buy* from the counterparty; CUAD also labels minimum sales efforts, sales-force sizes and
  royalty floors. Whether those belong in the category is a dataset-policy question.
- Some rows are fragments (a lead-in or a partial sentence) that carry no clause on their own.
