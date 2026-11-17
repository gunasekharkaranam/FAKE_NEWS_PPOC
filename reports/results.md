# Style model results (TF-IDF + Logistic Regression)

Label rule: **1 = fake, 0 = real**. Metric: **macro-F1** (fair when classes are unbalanced).

## Cross-dataset test (rows = trained on, columns = tested on)

| trained on \ tested on | liar | politifact | gossipcop |
|---|---|---|---|
| liar | **0.606** | 0.499 | 0.421 |
| politifact | 0.511 | **0.822** | 0.451 |
| gossipcop | 0.446 | 0.535 | **0.761** |

Bold = same dataset. Everything else = a dataset the model never saw.

## Final model (trained on all three)

| test set | accuracy | macro-F1 | fake-F1 | ROC-AUC | n |
|---|---|---|---|---|---|
| liar | 0.563 | 0.563 | 0.561 | 0.612 | 1283 |
| politifact | 0.628 | 0.627 | 0.618 | 0.736 | 196 |
| gossipcop | 0.82 | 0.761 | 0.643 | 0.855 | 4140 |

## Words the model links most with each class

- **Fake:** celebr react, beg, obama, trump, jealou, desper, report, brad, caitlyn, hollywoodlif, debunk, benghazi, brad pitt, obamacar, wisconsin
- **Real:** award, son, cardi, season, say donald, transcript, dress, wear, grace, sweet, recap, babi no, birthday, arrang, riverdal
