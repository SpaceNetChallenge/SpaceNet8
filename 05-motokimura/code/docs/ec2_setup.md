## Launch EC2 instance

Tested with `Deep Learning AMI GPU PyTorch 1.12.0 (Ubuntu 20.04) 20220803` and
`Deep Learning AMI GPU PyTorch 1.12.0 (Amazon Linux 2) 20220803` AMI.

## Download SpaceNet-8 dataset

Edit `~/.aws/credentials`.

```
mkdir ~/.aws
nano ~/.aws/credentials  # write your credentials at [default] section
```

Install AWS-CLI (can be skipped if Amazon Linux 2 AMI is used).

```
sudo apt install -y awscli
```

Prepare directories.

```
sudo mkdir -p /data/spacenet8/train
sudo mkdir -p /data/spacenet8/test
sudo mkdir -p /data/spacenet8_artifact
sudo chmod 777 -R /data
```

Download.

```
cd /data/spacenet8

aws s3 cp --recursive s3://spacenet-dataset/spacenet/SN8_floods/tarballs/ .
aws s3 cp s3://spacenet-dataset/spacenet/SN8_floods/resolutions.txt .

mkdir train/Germany_Training_Public
mv Germany_Training_Public.tar.gz train/Germany_Training_Public/
cd train/Germany_Training_Public/
tar -xvf Germany_Training_Public.tar.gz 
rm Germany_Training_Public.tar.gz

cd ../..
mkdir train/Louisiana-East_Training_Public
mv Louisiana-East_Training_Public.tar.gz train/Louisiana-East_Training_Public/
cd train/Louisiana-East_Training_Public/
tar -xvf Louisiana-East_Training_Public.tar.gz
rm Louisiana-East_Training_Public.tar.gz

cd ../..
mkdir test/Louisiana-West_Test_Public
mv Louisiana-West_Test_Public.tar.gz test/Louisiana-West_Test_Public/
cd test/Louisiana-West_Test_Public/
tar -xvf Louisiana-West_Test_Public.tar.gz
rm Louisiana-West_Test_Public.tar.gz

cd ../..
cp resolutions.txt train/
cp resolutions.txt test/
rm resolutions.txt
```

## Prepare solution

```
cd
git clone git@github.com:motokimura/spacenet8_solution.git
# or, git clone https://github.com/motokimura/spacenet8_solution.git
cd spacenet8_solution/
```

Go to [README.md](README.md).
