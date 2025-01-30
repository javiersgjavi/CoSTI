pip install gdown
gdown 1M_9uoUtLwYYXMcWmo8OsjoEsqv7HHHW9
unzip data_costi.zip
mv data_zip/weights/ ./
mkdir -p data/
sudo mv data_zip/mimic-iii_challenge/ data/
rmdir data_zip/