tr_df = train_df('/content/drive/MyDrive/Colab Notebooks/Osteo binary classification/OS Collected Data')
tr_df

train_df, dummies_df = train_test_split(tr_df, test_size=0.2, random_state=42)
valid_df, test_df = train_test_split(dummies_df, test_size=0.5, random_state=42)
