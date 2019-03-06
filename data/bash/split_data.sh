

if [ -d /home/datasets/mayilong/PycharmProjects/p55/data/split_data ]; then
        echo "Need to clean old files before split data"
        rm -rf /home/datasets/mayilong/PycharmProjects/p55/data/split_data
        echo "Clean Finish"
fi

echo "txt_dir = /home/datasets/mayilong/PycharmProjects/p55/data/all_data"
echo "save_dir= /home/datasets/mayilong/PycharmProjects/p55/data/split_data"
echo "video2path= /home/datasets/mayilong/PycharmProjects/p55/resource/train_val_video2path.pkl"

echo ""

echo "Start splitting"
python ./split_data.py \
--txt_dir='/home/datasets/mayilong/PycharmProjects/p55/data/all_data' \
--save_dir='/home/datasets/mayilong/PycharmProjects/p55/data/split_data' \
--video2path='/home/datasets/mayilong/PycharmProjects/p55/resource/train_val_video2path.pkl' \


echo "Finish splitting"
