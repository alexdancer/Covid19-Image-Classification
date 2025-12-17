## Notes on Covid19 Problems:

**12/9/2025**
- Having class imbalance issues so the model is not correctly predicting Indeterminate Appearance and Atypical Appearance at all
* I tried attention mechanism to fix the above problem, did not work very well

**12/12/2025**
- Still haven't fixed the class imbalance issues
- Added se_block function for Channel Attention and ResNet for slightly improved model
- Might have to get rid of Atypical Appearance and Indertiminate appearance images since the model cannot accurately predict those

**12/14/2025**
- Added model serialization to help save my model