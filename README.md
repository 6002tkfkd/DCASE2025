
## Test

  

### Evaluate on test set

To evaluate `log_loss` and `test_acc`:

  

```bash

python  main.py  test  --config  config/tfsepnet_test_han.yaml

```

  
  

## Predict

  

### How to Generate Submission CSV

  

To generate the submission CSV file, run the following command:

  

```bash

python  main.py  predict  --config  config/submit_csv.yaml

```
