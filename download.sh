mkdir data/full
wget https://zenodo.org/api/records/8275861/files-archive -O data/full/loghub-2.0.zip
unzip data/full/loghub-2.0.zip -d data/full
rm data/full/loghub-2.0.zip

# unzip large-scale datasets
#datasets=('BGL' 'HDFS' 'Linux' 'HealthApp' 'OpenStack' 'OpenSSH' 'Proxifier' 'HPC' 'Zookeeper' 'Mac' 'Hadoop' 'Apache' 'Thunderbird' 'Spark')
datasets=("HDFS" "BGL")
for dataset in ${datasets[@]}; do
    unzip "data/full/${dataset}.zip" -d "data/full"
    rm "data/full/${dataset}.zip"
done